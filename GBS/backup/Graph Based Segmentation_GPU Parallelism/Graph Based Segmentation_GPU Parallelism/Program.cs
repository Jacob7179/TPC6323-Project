using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

class Program
{
    // ================== CONFIGURATION ==================
    const string BaseFolderPath = @"C:\BrainTumorDataset\Testing\Dataset";
    const string OutputFolderPath = @"C:\BrainTumorDataset\Testing\GraphSegmented";
    const string GroundTruthMaskFolderPath = @"C:\BrainTumorDataset\Testing\Ground Truth Mask";

    const float K = 300f;
    const int MinSize = 100;
    const int BlurRadius = 2;

    // ===================================================

    // Thread‑safe locks
    private static readonly object consoleLock = new object();
    private static readonly object mseLock = new object();

    static void Main()
    {
        Directory.CreateDirectory(OutputFolderPath);

        // --------------------------------------------------
        // BALANCED DATASET SAMPLING (100 images total)
        // --------------------------------------------------
        string inputFolder = Path.GetFullPath(BaseFolderPath);

        string[] targetFolders = new[]
        {
            Path.Combine(inputFolder, "glioma"),
            Path.Combine(inputFolder, "pituitary"),
            Path.Combine(inputFolder, "meningioma")
        };

        var random = new Random(42);

        string[] glioma = Directory.GetFiles(targetFolders[0], "*", SearchOption.AllDirectories)
            .Where(IsSupportedImage)
            .OrderBy(_ => random.Next())
            .Take(33)
            .ToArray();

        string[] pituitary = Directory.GetFiles(targetFolders[1], "*", SearchOption.AllDirectories)
            .Where(IsSupportedImage)
            .OrderBy(_ => random.Next())
            .Take(33)
            .ToArray();

        string[] meningioma = Directory.GetFiles(targetFolders[2], "*", SearchOption.AllDirectories)
            .Where(IsSupportedImage)
            .OrderBy(_ => random.Next())
            .Take(34)
            .ToArray();

        string[] files = glioma
            .Concat(pituitary)
            .Concat(meningioma)
            .ToArray();

        Console.WriteLine($"Balanced dataset size: {files.Length}");

        string groundTruthFolder = ResolveGroundTruthFolder(GroundTruthMaskFolderPath);

        // --------------------------------------------------
        // PARALLEL PROCESSING (with GPU blur)
        // --------------------------------------------------
        RunParallelProcessing(files, groundTruthFolder);
    }

    // ------------------------------------------------------------
    //  PARALLEL PROCESSING
    // ------------------------------------------------------------
    static void RunParallelProcessing(string[] files, string groundTruthFolder)
    {
        if (files.Length == 0)
            throw new InvalidOperationException("No input images found.");

        Console.WriteLine($"Processing {files.Length} images in parallel (GPU blur) ...");
        Stopwatch swTotal = Stopwatch.StartNew();

        int processed = 0;
        var mseSummary = new MseSummary();

        var parallelOptions = new ParallelOptions
        {
            MaxDegreeOfParallelism = Environment.ProcessorCount // adjust as needed
        };

        Parallel.ForEach(files, parallelOptions, file =>
        {
            try
            {
                using Bitmap original = new Bitmap(file);
                using Bitmap rgb = ConvertTo24Bit(original);

                float[] gray = ToGray(rgb);
                gray = GpuBlur.Apply(gray, rgb.Width, rgb.Height, BlurRadius); // GPU accelerated
                int[] labels = GraphSegmentation(gray, rgb.Width, rgb.Height);
                int tumorLabel = BestTumorComponent(labels, gray, rgb.Width, rgb.Height);

                using Bitmap result = CreateGreenOverlay(rgb, labels, tumorLabel);

                string outFile = Path.Combine(
                    OutputFolderPath,
                    Path.GetFileNameWithoutExtension(file) + "_graph.png");
                result.Save(outFile, ImageFormat.Png);

                int current = Interlocked.Increment(ref processed);
                lock (consoleLock)
                {
                    Console.WriteLine($"Processed: {Path.GetFileName(file)} ({current}/{files.Length})");
                }

                // ----- MSE Evaluation (thread‑safe) -----
                bool[] predictedMask = new bool[gray.Length];
                for (int i = 0; i < gray.Length; i++)
                    predictedMask[i] = (labels[i] == tumorLabel);

                double? mse = EvaluateMseAgainstGroundTruth(
                    BaseFolderPath,
                    groundTruthFolder,
                    file,
                    rgb.Width,
                    rgb.Height,
                    predictedMask);

                lock (mseLock)
                {
                    mseSummary.Add(mse);
                }

                lock (consoleLock)
                {
                    Console.WriteLine(mse.HasValue
                        ? $"MSE: {mse.Value:F2}"
                        : "MSE: skipped (matching ground truth mask not found)");
                }
            }
            catch (Exception ex)
            {
                lock (consoleLock)
                {
                    Console.WriteLine($"Error: {file} -> {ex.Message}");
                }
            }
        });

        swTotal.Stop();

        Console.WriteLine();
        Console.WriteLine("================================");
        Console.WriteLine($"Images processed : {processed}");
        Console.WriteLine($"Total time       : {swTotal.Elapsed.TotalSeconds:F2}s");
        Console.WriteLine($"Average per image: {swTotal.Elapsed.TotalMilliseconds / processed:F2} ms");
        Console.WriteLine("================================");

        PrintMseSummary("GraphSeg", mseSummary);
    }

    // ============================================================
    //  MSE EVALUATION FUNCTIONS
    // ============================================================
    static string ResolveGroundTruthFolder(string groundTruthFolderPath)
    {
        if (string.IsNullOrWhiteSpace(groundTruthFolderPath))
            throw new ArgumentException("Please set GroundTruthMaskFolderPath in Program.cs.");

        string groundTruthFolder = Path.GetFullPath(groundTruthFolderPath);
        if (!Directory.Exists(groundTruthFolder))
            throw new DirectoryNotFoundException("Ground truth mask folder was not found: " + groundTruthFolder);

        return groundTruthFolder;
    }

    static double? EvaluateMseAgainstGroundTruth(
        string inputFolder,
        string groundTruthFolder,
        string inputPath,
        int width,
        int height,
        bool[] predictedMask)
    {
        string? groundTruthPath = FindGroundTruthMaskPath(inputFolder, groundTruthFolder, inputPath);
        if (groundTruthPath is null)
            return null;

        return CalculateMse(predictedMask, width, height, groundTruthPath);
    }

    static string? FindGroundTruthMaskPath(string inputFolder, string groundTruthFolder, string inputPath)
    {
        string relativeFolder = Path.GetDirectoryName(Path.GetRelativePath(inputFolder, inputPath)) ?? "";
        string maskFolder = Path.Combine(groundTruthFolder, relativeFolder);
        if (!Directory.Exists(maskFolder))
            return null;

        string stem = Path.GetFileNameWithoutExtension(inputPath);
        if (stem.EndsWith("_gray", StringComparison.OrdinalIgnoreCase))
            stem = stem[..^"_gray".Length];

        string[] candidateNames =
        {
            stem + "_gradcam",
            stem + "_mask",
            stem
        };

        string[] extensions = { ".png", ".jpg", ".jpeg", ".bmp" };
        foreach (string candidateName in candidateNames)
            foreach (string extension in extensions)
            {
                string candidatePath = Path.Combine(maskFolder, candidateName + extension);
                if (File.Exists(candidatePath))
                    return candidatePath;
            }

        return Directory
            .EnumerateFiles(maskFolder, "*.*", SearchOption.TopDirectoryOnly)
            .Where(IsSupportedImage)
            .FirstOrDefault(path => Path.GetFileNameWithoutExtension(path).StartsWith(stem, StringComparison.OrdinalIgnoreCase));
    }

    static double CalculateMse(bool[] predictedMask, int width, int height, string groundTruthPath)
    {
        using Bitmap source = new Bitmap(groundTruthPath);
        using Bitmap groundTruth = new Bitmap(width, height, PixelFormat.Format24bppRgb);
        using (Graphics graphics = Graphics.FromImage(groundTruth))
            graphics.DrawImage(source, new Rectangle(0, 0, width, height));

        BitmapData data = groundTruth.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
        double sumSquaredError = 0;

        unsafe
        {
            byte* ptr = (byte*)data.Scan0;
            for (int y = 0; y < height; y++)
            {
                byte* row = ptr + y * data.Stride;
                for (int x = 0; x < width; x++)
                {
                    int i = y * width + x, p = x * 3;
                    byte b = row[p], g = row[p + 1], r = row[p + 2];
                    double groundTruthValue = 0.299 * r + 0.587 * g + 0.114 * b;
                    double predictedValue = predictedMask[i] ? 255.0 : 0.0;
                    double diff = predictedValue - groundTruthValue;
                    sumSquaredError += diff * diff;
                }
            }
        }

        groundTruth.UnlockBits(data);
        return sumSquaredError / predictedMask.Length;
    }

    static void PrintMseSummary(string label, MseSummary summary)
    {
        Console.WriteLine();
        Console.WriteLine($"{label} MSE evaluation:");
        Console.WriteLine($"{label} MSE matched images: {summary.Count}");
        Console.WriteLine($"{label} MSE skipped images: {summary.Skipped}");
        if (summary.Count == 0)
            return;

        Console.WriteLine($"{label} MSE average: {summary.Average:F2}");
        Console.WriteLine($"{label} MSE min: {summary.Min:F2}");
        Console.WriteLine($"{label} MSE max: {summary.Max:F2}");
    }

    // ============================================================
    //  MSE SUMMARY CLASS
    // ============================================================
    class MseSummary
    {
        public double Total { get; private set; }
        public double Min { get; private set; } = double.MaxValue;
        public double Max { get; private set; }
        public int Count { get; private set; }
        public int Skipped { get; private set; }

        public double Average => Count == 0 ? 0 : Total / Count;

        public void Add(double? mse)
        {
            if (!mse.HasValue)
            {
                Skipped++;
                return;
            }

            Count++;
            Total += mse.Value;
            Min = Math.Min(Min, mse.Value);
            Max = Math.Max(Max, mse.Value);
        }
    }

    // ============================================================
    //  GPU BLUR (ILGPU) — Thread‑local accelerator for safety
    // ============================================================
    static class GpuBlur
    {
        // Each thread gets its own Context + Accelerator
        private class GpuResources : IDisposable
        {
            public Context Context { get; }
            public Accelerator Accelerator { get; }

            public GpuResources()
            {
                Context = Context.CreateDefault();
                Accelerator = Context.CreateCudaAccelerator(0);
            }

            public void Dispose()
            {
                Accelerator?.Dispose();
                Context?.Dispose();
            }
        }

        private static readonly ThreadLocal<GpuResources> _resources = new ThreadLocal<GpuResources>(
            () => new GpuResources(),
            trackAllValues: false
        );

        public static float[] Apply(float[] input, int width, int height, int radius)
        {
            // Get the accelerator for the current thread
            var resources = _resources.Value;
            var accelerator = resources.Accelerator;

            float[] kernel = BuildGaussianKernel(radius);
            using var kernelDev = accelerator.Allocate1D(kernel);
            using var inputDev = accelerator.Allocate1D(input);
            using var tempDev = accelerator.Allocate1D<float>(input.Length);
            using var outputDev = accelerator.Allocate1D<float>(input.Length);

            var horizKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, int, int, ArrayView<float>
            >(HorizontalBlur);

            var vertKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, int, int, ArrayView<float>
            >(VerticalBlur);

            int extent = input.Length;
            horizKernel(extent, inputDev.View, tempDev.View, width, height, kernelDev.View);
            accelerator.Synchronize();

            vertKernel(extent, tempDev.View, outputDev.View, width, height, kernelDev.View);
            accelerator.Synchronize();

            return outputDev.GetAsArray1D();
        }

        static float[] BuildGaussianKernel(int radius)
        {
            float sigma = radius / 2f;
            float[] kernel = new float[2 * radius + 1];
            float sum = 0;
            for (int i = -radius; i <= radius; i++)
            {
                float v = MathF.Exp(-(i * i) / (2 * sigma * sigma));
                kernel[i + radius] = v;
                sum += v;
            }
            for (int i = 0; i < kernel.Length; i++)
                kernel[i] /= sum;
            return kernel;
        }

        static void HorizontalBlur(
            Index1D index,
            ArrayView<float> input,
            ArrayView<float> output,
            int width,
            int height,
            ArrayView<float> kernel)
        {
            int idx = index.X;
            if (idx >= input.Length) return;

            int x = idx % width;
            int y = idx / width;
            int half = (int)(kernel.Length / 2);
            float sum = 0;

            for (int k = -half; k <= half; k++)
            {
                int xx = x + k;
                if (xx < 0) xx = 0;
                if (xx >= width) xx = width - 1;
                sum += input[y * width + xx] * kernel[k + half];
            }
            output[idx] = sum;
        }

        static void VerticalBlur(
            Index1D index,
            ArrayView<float> input,
            ArrayView<float> output,
            int width,
            int height,
            ArrayView<float> kernel)
        {
            int idx = index.X;
            if (idx >= input.Length) return;

            int x = idx % width;
            int y = idx / width;
            int half = (int)(kernel.Length / 2);
            float sum = 0;

            for (int k = -half; k <= half; k++)
            {
                int yy = y + k;
                if (yy < 0) yy = 0;
                if (yy >= height) yy = height - 1;
                sum += input[yy * width + x] * kernel[k + half];
            }
            output[idx] = sum;
        }
    }

    // ============================================================
    //  GRAPH SEGMENTATION (Felzenszwalb & Huttenlocher)
    // ============================================================
    struct Edge
    {
        public int A;
        public int B;
        public float Weight;
    }

    static int[] GraphSegmentation(float[] image, int width, int height)
    {
        int n = width * height;
        List<Edge> edges = BuildGraph(image, width, height);
        edges.Sort((a, b) => a.Weight.CompareTo(b.Weight));

        DisjointSet ds = new DisjointSet(n);
        float[] threshold = new float[n];
        for (int i = 0; i < n; i++)
            threshold[i] = K;

        foreach (var e in edges)
        {
            int a = ds.Find(e.A);
            int b = ds.Find(e.B);
            if (a == b) continue;

            if (e.Weight <= threshold[a] && e.Weight <= threshold[b])
            {
                ds.Union(a, b);
                int root = ds.Find(a);
                threshold[root] = e.Weight + K / ds.Size(root);
            }
        }

        foreach (var e in edges)
        {
            int a = ds.Find(e.A);
            int b = ds.Find(e.B);
            if (a == b) continue;

            if (ds.Size(a) < MinSize || ds.Size(b) < MinSize)
                ds.Union(a, b);
        }

        int[] labels = new int[n];
        for (int i = 0; i < n; i++)
            labels[i] = ds.Find(i);

        return labels;
    }

    static List<Edge> BuildGraph(float[] image, int width, int height)
    {
        List<Edge> edges = new();
        for (int y = 0; y < height; y++)
        {
            int row = y * width;
            for (int x = 0; x < width; x++)
            {
                int p = row + x;

                if (x < width - 1)
                    edges.Add(new Edge { A = p, B = p + 1, Weight = MathF.Abs(image[p] - image[p + 1]) });

                if (y < height - 1)
                    edges.Add(new Edge { A = p, B = p + width, Weight = MathF.Abs(image[p] - image[p + width]) });

                if (x < width - 1 && y < height - 1)
                    edges.Add(new Edge { A = p, B = p + width + 1, Weight = MathF.Abs(image[p] - image[p + width + 1]) });

                if (x < width - 1 && y > 0)
                    edges.Add(new Edge { A = p, B = p - width + 1, Weight = MathF.Abs(image[p] - image[p - width + 1]) });
            }
        }
        return edges;
    }

    // ============================================================
    //  UNION-FIND
    // ============================================================
    class DisjointSet
    {
        int[] parent;
        int[] size;

        public DisjointSet(int n)
        {
            parent = new int[n];
            size = new int[n];
            for (int i = 0; i < n; i++)
            {
                parent[i] = i;
                size[i] = 1;
            }
        }

        public int Find(int x)
        {
            if (parent[x] == x) return x;
            parent[x] = Find(parent[x]);
            return parent[x];
        }

        public void Union(int a, int b)
        {
            a = Find(a);
            b = Find(b);
            if (a == b) return;
            if (size[a] < size[b]) (a, b) = (b, a);
            parent[b] = a;
            size[a] += size[b];
        }

        public int Size(int x) => size[Find(x)];
    }

    // ============================================================
    //  TUMOR COMPONENT SELECTION
    // ============================================================
    static int BestTumorComponent(int[] labels, float[] gray, int width, int height)
    {
        int n = labels.Length;
        var stats = new Dictionary<int, (int count, float sum, int minX, int maxX, int minY, int maxY)>();

        for (int i = 0; i < n; i++)
        {
            int l = labels[i];
            if (!stats.ContainsKey(l))
                stats[l] = (0, 0f, width, -1, height, -1);

            var (c, s, minX, maxX, minY, maxY) = stats[l];
            int x = i % width;
            int y = i / width;
            stats[l] = (c + 1, s + gray[i],
                Math.Min(minX, x), Math.Max(maxX, x),
                Math.Min(minY, y), Math.Max(maxY, y));
        }

        bool TouchesBorder(int label)
        {
            var (_, _, minX, maxX, minY, maxY) = stats[label];
            return minX == 0 || maxX == width - 1 || minY == 0 || maxY == height - 1;
        }

        int brainLabel = -1;
        int maxBrainSize = 0;
        foreach (var kv in stats)
        {
            int label = kv.Key;
            var (count, sum, _, _, _, _) = kv.Value;
            float mean = sum / count;
            if (!TouchesBorder(label) && mean > 0.2f && mean < 0.8f)
            {
                if (count > maxBrainSize)
                {
                    maxBrainSize = count;
                    brainLabel = label;
                }
            }
        }

        bool hasBrain = brainLabel != -1;

        float globalMean = 0f;
        foreach (var v in stats.Values) globalMean += v.sum;
        globalMean /= n;

        var scores = new List<(int label, float score)>();

        foreach (var kv in stats)
        {
            int label = kv.Key;
            var (count, sum, minX, maxX, minY, maxY) = kv.Value;

            if (hasBrain && label == brainLabel) continue;

            if (hasBrain)
            {
                if (TouchesBorder(label)) continue;

                int cx = 0, cy = 0;
                for (int i = 0; i < n; i++)
                    if (labels[i] == label)
                    {
                        int x = i % width;
                        int y = i / width;
                        cx += x;
                        cy += y;
                    }
                cx /= count;
                cy /= count;

                var (_, _, bMinX, bMaxX, bMinY, bMaxY) = stats[brainLabel];
                if (cx < bMinX || cx > bMaxX || cy < bMinY || cy > bMaxY)
                    continue;
            }
            else
            {
                if (TouchesBorder(label)) continue;
            }

            float mean = sum / count;

            float intensityScore = mean - globalMean;
            if (intensityScore < 0) intensityScore *= 0.2f;

            float sizeScore = MathF.Log(count + 1) * 0.5f;

            float perimeter = 0f;
            int ww = maxX - minX + 1;
            int hh = maxY - minY + 1;
            bool[,] mask = new bool[ww, hh];
            for (int y = minY; y <= maxY; y++)
                for (int x = minX; x <= maxX; x++)
                    if (labels[y * width + x] == label)
                        mask[x - minX, y - minY] = true;

            for (int y = 0; y < hh; y++)
                for (int x = 0; x < ww; x++)
                    if (mask[x, y])
                    {
                        if (x == 0 || !mask[x - 1, y]) perimeter++;
                        if (x == ww - 1 || !mask[x + 1, y]) perimeter++;
                        if (y == 0 || !mask[x, y - 1]) perimeter++;
                        if (y == hh - 1 || !mask[x, y + 1]) perimeter++;
                    }
            float circularity = (4 * MathF.PI * count) / (perimeter * perimeter + 1e-6f);
            float shapeScore = MathF.Min(circularity, 1.0f);

            float finalScore = intensityScore * 1.2f + sizeScore * 0.3f + shapeScore * 0.5f;
            scores.Add((label, finalScore));
        }

        if (scores.Count == 0)
            return stats.OrderByDescending(kv => kv.Value.count).First().Key;

        return scores.OrderByDescending(s => s.score).First().label;
    }

    // ============================================================
    //  IMAGE UTILITIES
    // ============================================================
    static bool IsSupportedImage(string path)
    {
        string ext = Path.GetExtension(path).ToLower();
        return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
    }

    static Bitmap ConvertTo24Bit(Bitmap src)
    {
        Bitmap bmp = new Bitmap(src.Width, src.Height, PixelFormat.Format24bppRgb);
        using Graphics g = Graphics.FromImage(bmp);
        g.DrawImage(src, 0, 0);
        return bmp;
    }

    static unsafe float[] ToGray(Bitmap bmp)
    {
        int w = bmp.Width;
        int h = bmp.Height;
        float[] gray = new float[w * h];

        BitmapData data = bmp.LockBits(
            new Rectangle(0, 0, w, h),
            ImageLockMode.ReadOnly,
            PixelFormat.Format24bppRgb);

        byte* ptr = (byte*)data.Scan0;
        for (int y = 0; y < h; y++)
        {
            byte* row = ptr + y * data.Stride;
            for (int x = 0; x < w; x++)
            {
                int p = x * 3;
                gray[y * w + x] = 0.299f * row[p + 2] + 0.587f * row[p + 1] + 0.114f * row[p];
            }
        }
        bmp.UnlockBits(data);
        return gray;
    }

    static unsafe Bitmap CreateGreenOverlay(Bitmap original, int[] labels, int tumorLabel)
    {
        int w = original.Width;
        int h = original.Height;
        Bitmap result = new Bitmap(w, h, PixelFormat.Format24bppRgb);

        BitmapData src = original.LockBits(
            new Rectangle(0, 0, w, h),
            ImageLockMode.ReadOnly,
            PixelFormat.Format24bppRgb);

        BitmapData dst = result.LockBits(
            new Rectangle(0, 0, w, h),
            ImageLockMode.WriteOnly,
            PixelFormat.Format24bppRgb);

        byte* sp = (byte*)src.Scan0;
        byte* dp = (byte*)dst.Scan0;

        for (int y = 0; y < h; y++)
        {
            byte* srow = sp + y * src.Stride;
            byte* drow = dp + y * dst.Stride;
            for (int x = 0; x < w; x++)
            {
                int p = x * 3;
                int idx = y * w + x;
                if (labels[idx] == tumorLabel)
                {
                    drow[p] = 0;
                    drow[p + 1] = 255;
                    drow[p + 2] = 0;
                }
                else
                {
                    byte g = (byte)(0.299f * srow[p + 2] + 0.587f * srow[p + 1] + 0.114f * srow[p]);
                    drow[p] = g;
                    drow[p + 1] = g;
                    drow[p + 2] = g;
                }
            }
        }

        original.UnlockBits(src);
        result.UnlockBits(dst);
        return result;
    }
}