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
    const string BaseFolderPath = @"C:\BrainTumorDataset\Testing\Testing";
    const string OutputFolderPath = @"C:\BrainTumorDataset\Testing\GraphSegmented";

    const float K = 300f;
    const int MinSize = 100;
    const int BlurRadius = 2;

    static void Main()
    {
        Directory.CreateDirectory(OutputFolderPath);

        string[] files = Directory
            .EnumerateFiles(BaseFolderPath, "*.*", SearchOption.AllDirectories)
            .Where(IsSupportedImage)
            .ToArray();

        Console.WriteLine($"Found {files.Length} images");

        Stopwatch sw = Stopwatch.StartNew();

        int total = 0;
        object consoleLock = new();

        Parallel.ForEach(
            files,
            new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
            file =>
            {
                try
                {
                    using Bitmap original = new Bitmap(file);
                    using Bitmap rgb = ConvertTo24Bit(original);

                    float[] gray = ToGray(rgb);

                    // GPU‑accelerated Gaussian blur
                    gray = GpuBlur.Apply(gray, rgb.Width, rgb.Height, BlurRadius);

                    int[] labels = GraphSegmentation(gray, rgb.Width, rgb.Height);
                    int tumorLabel = BestTumorComponent(labels, gray, rgb.Width, rgb.Height);

                    using Bitmap result = CreateGreenOverlay(rgb, labels, tumorLabel);

                    string outFile = Path.Combine(
                        OutputFolderPath,
                        Path.GetFileNameWithoutExtension(file) + "_graph.png");

                    result.Save(outFile, ImageFormat.Png);

                    Interlocked.Increment(ref total);

                    lock (consoleLock)
                    {
                        Console.WriteLine($"Processed: {Path.GetFileName(file)}");
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

        sw.Stop();

        Console.WriteLine();
        Console.WriteLine("================================");
        Console.WriteLine($"Images : {total}");
        Console.WriteLine($"Time   : {sw.Elapsed.TotalSeconds:F2}s");
        Console.WriteLine($"Avg    : {sw.Elapsed.TotalMilliseconds / total:F2} ms");
        Console.WriteLine("================================");
    }

    // ============================================================
    //  GPU BLUR (ILGPU) — FIXED: manual clamping
    // ============================================================
    static class GpuBlur
    {
        public static float[] Apply(float[] input, int width, int height, int radius)
        {
            using var context = Context.CreateDefault();
            using var accelerator = context.CreateCudaAccelerator(0);

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
                // manual clamp
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
                // manual clamp
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