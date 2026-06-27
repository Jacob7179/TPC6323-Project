using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

class Program
{
    // ================== CONFIGURATION ==================
    const string BaseFolderPath = @"C:\BrainTumorDataset\Testing\Dataset";
    const string OutputFolderPath = @"C:\BrainTumorDataset\Testing\GraphSegmented";
    const string GroundTruthMaskFolderPath = @"C:\BrainTumorDataset\Testing\Ground Truth Mask";
    const int MinBrainRoiPixels = 100;
    const float MinCandidateComponentRatio = 0.00035f;
    const float MaxCandidateComponentRatio = 0.18f;
    const float MaxBrainBoundaryTouchRatio = 0.10f;

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

        var random = new Random(42); // fixed seed for reproducibility

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

        // Resolve ground truth folder
        string groundTruthFolder = ResolveGroundTruthFolder(GroundTruthMaskFolderPath);

        // --------------------------------------------------
        // PARALLEL PROCESSING
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

        Console.WriteLine($"Processing {files.Length} images in parallel...");
        Stopwatch swTotal = Stopwatch.StartNew();

        int processed = 0;
        var mseSummary = new MseSummary();

        // Configure parallelism – you can change MaxDegreeOfParallelism
        var parallelOptions = new ParallelOptions
        {
            MaxDegreeOfParallelism = Environment.ProcessorCount // or a fixed number like 4
        };

        Parallel.ForEach(files, parallelOptions, file =>
        {
            try
            {
                // ---------- Process one image ----------
                using Bitmap original = new Bitmap(file);
                using Bitmap rgb = ConvertTo24Bit(original);

                float[] gray = ToGray(rgb);
                gray = GaussianBlur(gray, rgb.Width, rgb.Height, BlurRadius);
                int[] labels = GraphSegmentation(gray, rgb.Width, rgb.Height);
                int tumorLabel = BestTumorComponent(labels, gray, rgb.Width, rgb.Height);

                bool[] tumorMask = BuildSingleLabelCandidateMask(labels, tumorLabel, gray, rgb.Width, rgb.Height);
                using Bitmap result = CreateGreenOverlay(rgb, tumorMask);

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

        // Print MSE summary
        PrintMseSummary("GraphSeg", mseSummary);
    }

    // ============================================================
    //  MSE EVALUATION FUNCTIONS (unchanged)
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
    //  GRAPH SEGMENTATION (Felzenszwalb & Huttenlocher) — CPU
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
            if (a == b)
                continue;
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
            if (a == b)
                continue;
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
            if (parent[x] == x)
                return x;
            parent[x] = Find(parent[x]);
            return parent[x];
        }

        public void Union(int a, int b)
        {
            a = Find(a);
            b = Find(b);
            if (a == b)
                return;
            if (size[a] < size[b])
            {
                int t = a;
                a = b;
                b = t;
            }
            parent[b] = a;
            size[a] += size[b];
        }

        public int Size(int x) => size[Find(x)];
    }

    // ============================================================
    //  TUMOR COMPONENT SELECTION (unchanged)
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

    static bool[] BuildBrainRoiMask(float[] gray, int w, int h)
    {
        bool[] foreground = ThresholdForeground(gray);
        foreground = CloseMask(foreground, w, h, 2);
        foreground = KeepLargestComponent(foreground, w, h);
        FillHolesInPlace(foreground, w, h);
        ApplyCranialPriorMask(foreground, w, h);

        int erosionRadius = Math.Clamp(Math.Min(w, h) / 38, 8, 18);
        bool[] eroded = ErodeMask(foreground, w, h, erosionRadius);
        eroded = KeepLargestComponent(eroded, w, h);
        if (CountMask(eroded) >= MinBrainRoiPixels)
            return eroded;

        int fallbackRadius = Math.Clamp(Math.Min(w, h) / 55, 5, 12);
        bool[] fallback = ErodeMask(foreground, w, h, fallbackRadius);
        fallback = KeepLargestComponent(fallback, w, h);
        return CountMask(fallback) >= MinBrainRoiPixels ? fallback : new bool[w * h];
    }

    static bool[] ThresholdForeground(float[] gray)
    {
        float threshold = Math.Clamp(EstimateForegroundThreshold(gray) * 0.35f, 6f, 45f);
        bool[] mask = new bool[gray.Length];

        for (int i = 0; i < gray.Length; i++)
            mask[i] = gray[i] > threshold;

        return mask;
    }

    static int EstimateForegroundThreshold(float[] gray)
    {
        int[] histogram = new int[256];
        foreach (float value in gray)
        {
            int bucket = Math.Clamp((int)MathF.Round(value), 0, 255);
            histogram[bucket]++;
        }

        long total = gray.Length;
        double sum = 0;
        for (int i = 0; i < histogram.Length; i++)
            sum += i * histogram[i];

        long backgroundWeight = 0;
        double backgroundSum = 0;
        double bestVariance = 0;
        int bestThreshold = 0;

        for (int threshold = 0; threshold < histogram.Length; threshold++)
        {
            backgroundWeight += histogram[threshold];
            if (backgroundWeight == 0)
                continue;

            long foregroundWeight = total - backgroundWeight;
            if (foregroundWeight == 0)
                break;

            backgroundSum += threshold * histogram[threshold];
            double backgroundMean = backgroundSum / backgroundWeight;
            double foregroundMean = (sum - backgroundSum) / foregroundWeight;
            double difference = backgroundMean - foregroundMean;
            double variance = (double)backgroundWeight * foregroundWeight * difference * difference;

            if (variance > bestVariance)
            {
                bestVariance = variance;
                bestThreshold = threshold;
            }
        }

        return bestThreshold;
    }

    static void ApplyCranialPriorMask(bool[] mask, int w, int h)
    {
        float centerX = w * 0.45f;
        float centerY = h * 0.44f;
        float radiusX = w * 0.41f;
        float radiusY = h * 0.39f;
        float radiusX2 = radiusX * radiusX;
        float radiusY2 = radiusY * radiusY;
        int lowerLimit = (int)(h * 0.78f);

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                int index = y * w + x;
                if (!mask[index])
                    continue;

                float dx = x - centerX;
                float dy = y - centerY;
                bool insideCranialArea = y <= lowerLimit && (dx * dx / radiusX2 + dy * dy / radiusY2) <= 1f;
                if (!insideCranialArea)
                    mask[index] = false;
            }
        }
    }

    static bool[] BuildTumorCandidateMask(int[] labels, float[] centroids, bool[] brainMask, int w, int h)
    {
        int tumorCluster = FindTumorClusterInBrainRoi(labels, centroids, brainMask, w, h);
        bool[] rawCandidate = new bool[w * h];
        if (tumorCluster < 0)
            return rawCandidate;

        for (int i = 0; i < rawCandidate.Length; i++)
            rawCandidate[i] = brainMask[i] && labels[i] == tumorCluster;

        return CleanCandidateComponents(rawCandidate, brainMask, w, h);
    }

    static bool[] BuildSingleLabelCandidateMask(int[] labels, int tumorLabel, float[] gray, int w, int h)
    {
        bool[] brainMask = BuildBrainRoiMask(gray, w, h);
        bool[] rawCandidate = new bool[w * h];

        for (int i = 0; i < rawCandidate.Length; i++)
            rawCandidate[i] = brainMask[i] && labels[i] == tumorLabel;

        return CleanCandidateComponents(rawCandidate, brainMask, w, h);
    }

    static int FindTumorClusterInBrainRoi(int[] labels, float[] centroids, bool[] brainMask, int w, int h)
    {
        int k = centroids.Length;
        int[] counts = new int[k];
        int[] boundaryCounts = new int[k];
        int roiCount = 0;

        for (int i = 0; i < labels.Length; i++)
        {
            if (!brainMask[i])
                continue;

            roiCount++;
            int label = labels[i];
            if (label < 0 || label >= k)
                continue;

            counts[label]++;
            if (IsBrainBoundaryPixel(brainMask, w, h, i))
                boundaryCounts[label]++;
        }

        if (roiCount < MinBrainRoiPixels)
            return -1;

        int minClusterPixels = Math.Max(8, (int)(roiCount * 0.002f));
        int best = -1;
        float bestScore = float.MinValue;

        for (int c = 0; c < k; c++)
        {
            if (counts[c] < minClusterPixels)
                continue;

            float sizeRatio = counts[c] / (float)roiCount;
            float boundaryRatio = boundaryCounts[c] / (float)counts[c];
            float score = centroids[c] - sizeRatio * 95f - boundaryRatio * 80f;

            if (sizeRatio > 0.45f)
                score -= 120f;

            if (score > bestScore)
            {
                bestScore = score;
                best = c;
            }
        }

        return best;
    }

    static bool[] CleanCandidateComponents(bool[] candidate, bool[] brainMask, int w, int h)
    {
        int n = candidate.Length;
        bool[] result = new bool[n];
        bool[] visited = new bool[n];
        int[] queue = new int[n];
        int[] current = new int[n];
        int[] bestComponent = new int[n];
        int bestCount = 0;
        float bestScore = float.MinValue;

        int roiCount = CountMask(brainMask);
        int minSize = Math.Max(12, (int)(roiCount * MinCandidateComponentRatio));
        int maxSize = Math.Max(minSize + 1, (int)(roiCount * MaxCandidateComponentRatio));

        for (int start = 0; start < n; start++)
        {
            if (!candidate[start] || visited[start])
                continue;

            int head = 0;
            int tail = 0;
            int count = 0;
            int boundaryTouches = 0;
            long sumY = 0;
            bool touchesImageBorder = false;

            queue[tail++] = start;
            visited[start] = true;

            while (head < tail)
            {
                int index = queue[head++];
                current[count++] = index;

                int x = index % w;
                int y = index / w;
                sumY += y;
                touchesImageBorder |= x == 0 || y == 0 || x == w - 1 || y == h - 1;

                if (IsBrainBoundaryPixel(brainMask, w, h, index))
                    boundaryTouches++;

                for (int dy = -1; dy <= 1; dy++)
                {
                    int yy = y + dy;
                    if (yy < 0 || yy >= h)
                        continue;

                    for (int dx = -1; dx <= 1; dx++)
                    {
                        if (dx == 0 && dy == 0)
                            continue;

                        int xx = x + dx;
                        if (xx < 0 || xx >= w)
                            continue;

                        int next = yy * w + xx;
                        if (candidate[next] && !visited[next])
                        {
                            visited[next] = true;
                            queue[tail++] = next;
                        }
                    }
                }
            }

            float boundaryTouchRatio = boundaryTouches / (float)count;
            bool keepComponent =
                count >= minSize &&
                count <= maxSize &&
                !touchesImageBorder &&
                boundaryTouchRatio <= MaxBrainBoundaryTouchRatio;

            if (!keepComponent)
                continue;

            float centerYRatio = (sumY / (float)count) / h;
            float lowerRegionPenalty = Math.Max(0f, centerYRatio - 0.62f) * count * 1.8f;
            float score = count * (1f - boundaryTouchRatio) - lowerRegionPenalty;

            if (score > bestScore)
            {
                bestScore = score;
                bestCount = count;
                Array.Copy(current, bestComponent, count);
            }
        }

        for (int i = 0; i < bestCount; i++)
            result[bestComponent[i]] = true;

        return result;
    }

    static int CountMask(bool[] mask)
    {
        int count = 0;
        foreach (bool value in mask)
        {
            if (value)
                count++;
        }

        return count;
    }

    static bool[] CloseMask(bool[] mask, int w, int h, int radius)
    {
        if (radius <= 0)
            return (bool[])mask.Clone();

        return ErodeMask(DilateMask(mask, w, h, radius), w, h, radius);
    }

    static bool[] DilateMask(bool[] mask, int w, int h, int radius)
    {
        bool[] result = new bool[mask.Length];

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                bool value = false;

                for (int dy = -radius; dy <= radius && !value; dy++)
                {
                    int yy = y + dy;
                    if (yy < 0 || yy >= h)
                        continue;

                    for (int dx = -radius; dx <= radius; dx++)
                    {
                        int xx = x + dx;
                        if (xx < 0 || xx >= w)
                            continue;

                        if (mask[yy * w + xx])
                        {
                            value = true;
                            break;
                        }
                    }
                }

                result[y * w + x] = value;
            }
        }

        return result;
    }

    static bool[] ErodeMask(bool[] mask, int w, int h, int radius)
    {
        if (radius <= 0)
            return (bool[])mask.Clone();

        bool[] result = new bool[mask.Length];

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                bool keep = true;

                for (int dy = -radius; dy <= radius && keep; dy++)
                {
                    int yy = y + dy;
                    if (yy < 0 || yy >= h)
                    {
                        keep = false;
                        break;
                    }

                    for (int dx = -radius; dx <= radius; dx++)
                    {
                        int xx = x + dx;
                        if (xx < 0 || xx >= w || !mask[yy * w + xx])
                        {
                            keep = false;
                            break;
                        }
                    }
                }

                result[y * w + x] = keep;
            }
        }

        return result;
    }

    static bool[] KeepLargestComponent(bool[] mask, int w, int h)
    {
        int n = mask.Length;
        bool[] visited = new bool[n];
        int[] queue = new int[n];
        int[] current = new int[n];
        int[] bestComponent = new int[n];
        int bestCount = 0;

        for (int start = 0; start < n; start++)
        {
            if (!mask[start] || visited[start])
                continue;

            int head = 0;
            int tail = 0;
            int count = 0;

            queue[tail++] = start;
            visited[start] = true;

            while (head < tail)
            {
                int index = queue[head++];
                current[count++] = index;

                int x = index % w;
                int y = index / w;

                for (int dy = -1; dy <= 1; dy++)
                {
                    int yy = y + dy;
                    if (yy < 0 || yy >= h)
                        continue;

                    for (int dx = -1; dx <= 1; dx++)
                    {
                        if (dx == 0 && dy == 0)
                            continue;

                        int xx = x + dx;
                        if (xx < 0 || xx >= w)
                            continue;

                        int next = yy * w + xx;
                        if (mask[next] && !visited[next])
                        {
                            visited[next] = true;
                            queue[tail++] = next;
                        }
                    }
                }
            }

            if (count > bestCount)
            {
                bestCount = count;
                Array.Copy(current, bestComponent, count);
            }
        }

        bool[] result = new bool[n];
        for (int i = 0; i < bestCount; i++)
            result[bestComponent[i]] = true;

        return result;
    }

    static void FillHolesInPlace(bool[] mask, int w, int h)
    {
        bool[] outside = new bool[mask.Length];
        int[] queue = new int[mask.Length];
        int head = 0;
        int tail = 0;

        void AddOutsidePixel(int index)
        {
            if (!mask[index] && !outside[index])
            {
                outside[index] = true;
                queue[tail++] = index;
            }
        }

        for (int x = 0; x < w; x++)
        {
            AddOutsidePixel(x);
            AddOutsidePixel((h - 1) * w + x);
        }

        for (int y = 0; y < h; y++)
        {
            AddOutsidePixel(y * w);
            AddOutsidePixel(y * w + w - 1);
        }

        while (head < tail)
        {
            int index = queue[head++];
            int x = index % w;
            int y = index / w;

            if (x > 0) AddOutsidePixel(index - 1);
            if (x < w - 1) AddOutsidePixel(index + 1);
            if (y > 0) AddOutsidePixel(index - w);
            if (y < h - 1) AddOutsidePixel(index + w);
        }

        for (int i = 0; i < mask.Length; i++)
        {
            if (!mask[i] && !outside[i])
                mask[i] = true;
        }
    }

    static bool IsBrainBoundaryPixel(bool[] brainMask, int w, int h, int index)
    {
        if (!brainMask[index])
            return true;

        int x = index % w;
        int y = index / w;

        if (x == 0 || y == 0 || x == w - 1 || y == h - 1)
            return true;

        return
            !brainMask[index - 1] ||
            !brainMask[index + 1] ||
            !brainMask[index - w] ||
            !brainMask[index + w];
    }

    // ============================================================
    //  IMAGE UTILITIES (unchanged)
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

        BitmapData data = bmp.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
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

    static float[] GaussianBlur(float[] input, int w, int h, int r)
    {
        float[] temp = new float[input.Length];
        float[] output = new float[input.Length];

        float sigma = r / 2f;
        float[] kernel = new float[r * 2 + 1];
        float sum = 0;

        for (int i = -r; i <= r; i++)
        {
            float v = MathF.Exp(-(i * i) / (2 * sigma * sigma));
            kernel[i + r] = v;
            sum += v;
        }

        for (int i = 0; i < kernel.Length; i++)
            kernel[i] /= sum;

        for (int y = 0; y < h; y++)
        {
            int row = y * w;
            for (int x = 0; x < w; x++)
            {
                float s = 0;
                for (int k = -r; k <= r; k++)
                {
                    int xx = Math.Clamp(x + k, 0, w - 1);
                    s += input[row + xx] * kernel[k + r];
                }
                temp[row + x] = s;
            }
        }

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                float s = 0;
                for (int k = -r; k <= r; k++)
                {
                    int yy = Math.Clamp(y + k, 0, h - 1);
                    s += temp[yy * w + x] * kernel[k + r];
                }
                output[y * w + x] = s;
            }
        }

        return output;
    }

    static unsafe Bitmap CreateGreenOverlay(Bitmap original, bool[] highlightMask)
    {
        int w = original.Width;
        int h = original.Height;

        Bitmap result = new Bitmap(w, h, PixelFormat.Format24bppRgb);

        BitmapData src = original.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
        BitmapData dst = result.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);

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

                if (highlightMask[idx])
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