using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
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
    const string BaseFolderPath = @"C:\BrainTumorDataset\Testing\Testing";
    const string OutputFolderPath = @"C:\BrainTumorDataset\Testing\GraphSegmented";

    // Sequential Graph (Felzenszwalb) parameters
    const float K = 300f;
    const int MinSize = 100;
    const int BlurRadius = 2;          // set to 0 to disable blur

    // Graph Cut parameters
    const float Lambda = 0.5f;
    const float Sigma = 30f;

    // Common parameters for GNN methods (superpixel graph)
    const int SuperpixelCount = 500;
    const int SlicIterations = 8;
    const float SlicCompactness = 10f;

    const bool RunBenchmarkDotNet = false;
    const string BenchmarkLabel = "GraphSeg";

    // Select method: 0=SequentialGraph, 1=GraphCut, 2=GCN, 3=HGNN, 4=GAT
    const int Method = 0;

    // Path to ONNX models (for methods 2‑4)
    const string GcnModelPath = @"models/gcn.onnx";
    const string HgnnModelPath = @"models/hgnn.onnx";
    const string GatModelPath = @"models/gat.onnx";

    // ===================================================

    static void Main(string[] args)
    {
        bool runOnce = args.Contains("--run-once", StringComparer.OrdinalIgnoreCase);
        bool runBenchmark = args.Contains("--benchmark", StringComparer.OrdinalIgnoreCase);

        if ((RunBenchmarkDotNet && !runOnce) || args.Contains("--benchmark-dotnet", StringComparer.OrdinalIgnoreCase))
        {
            BenchmarkRunner.Run<GraphSegBenchmark>();
            return;
        }

        string inputFolder = Path.GetFullPath(BaseFolderPath);
        Directory.CreateDirectory(OutputFolderPath);

        string[] targetFolders = new[]
        {
            Path.Combine(inputFolder, "glioma"),
            Path.Combine(inputFolder, "pituitary"),
            Path.Combine(inputFolder, "meningioma")
        };

        var random = new Random(42);
        string[] glioma = Directory.GetFiles(targetFolders[0], "*", SearchOption.AllDirectories)
            .Where(IsSupportedImage).OrderBy(_ => random.Next()).Take(33).ToArray();
        string[] pituitary = Directory.GetFiles(targetFolders[1], "*", SearchOption.AllDirectories)
            .Where(IsSupportedImage).OrderBy(_ => random.Next()).Take(33).ToArray();
        string[] meningioma = Directory.GetFiles(targetFolders[2], "*", SearchOption.AllDirectories)
            .Where(IsSupportedImage).OrderBy(_ => random.Next()).Take(34).ToArray();

        string[] files = glioma.Concat(pituitary).Concat(meningioma).ToArray();
        Console.WriteLine($"Balanced dataset size: {files.Length}");

        if (runBenchmark || args.Contains("--benchmark", StringComparer.OrdinalIgnoreCase))
        {
            RunManualBenchmark(files);
            return;
        }

        RunNormalProcessing(files);
    }

    // ------------------------------------------------------------
    //  NORMAL PARALLEL PROCESSING
    // ------------------------------------------------------------
    static void RunNormalProcessing(string[] files)
    {
        Stopwatch swTotal = Stopwatch.StartNew();
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

                    // Pre‑processing (optional blur)
                    if (Method == 0 && BlurRadius > 0)
                        gray = ApplyGaussianBlur(gray, rgb.Width, rgb.Height, BlurRadius);

                    int[] labels = null;
                    int tumorLabel = -1;

                    switch (Method)
                    {
                        case 0:
                            labels = GraphSegmentation(gray, rgb.Width, rgb.Height);
                            tumorLabel = BestTumorComponent(labels, gray, rgb.Width, rgb.Height);
                            break;
                        case 1:
                            labels = GraphCutSegmentation(gray, rgb.Width, rgb.Height);
                            tumorLabel = 1;
                            break;
                        case 2:
                        case 3:
                        case 4:
                            (labels, tumorLabel) = GnnSegmentation(gray, rgb.Width, rgb.Height, Method);
                            break;
                    }

                    using Bitmap result = CreateOverlay(rgb, labels, tumorLabel, Method);
                    string outFile = Path.Combine(OutputFolderPath,
                        Path.GetFileNameWithoutExtension(file) + "_" + GetMethodName(Method) + ".png");
                    result.Save(outFile, ImageFormat.Png);

                    Interlocked.Increment(ref total);
                    lock (consoleLock) Console.WriteLine($"Processed: {Path.GetFileName(file)}");
                }
                catch (Exception ex)
                {
                    lock (consoleLock) Console.WriteLine($"Error: {file} -> {ex.Message}");
                }
            });

        swTotal.Stop();
        Console.WriteLine($"\nTotal images: {total}");
        Console.WriteLine($"Total time: {swTotal.Elapsed.TotalSeconds:F2}s");
        Console.WriteLine($"Avg time: {swTotal.Elapsed.TotalMilliseconds / total:F2} ms");
    }

    // ------------------------------------------------------------
    //  MANUAL BENCHMARK (SEQUENTIAL)
    // ------------------------------------------------------------
    static void RunManualBenchmark(string[] files)
    {
        if (files.Length == 0) throw new InvalidOperationException("No input images found.");
        Console.WriteLine($"Benchmark: {files.Length} images (sequential)");

        var summary = new BenchmarkSummary();
        var swTotal = Stopwatch.StartNew();

        foreach (string file in files)
        {
            try
            {
                var result = ProcessOneImage(file);
                summary.Add(result.Timing);
                Console.WriteLine($"Processed: {Path.GetFileName(file)} -> {result.OutputPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {file} -> {ex.Message}");
            }
        }

        swTotal.Stop();
        PrintBenchmarkSummary(BenchmarkLabel, summary, swTotal.Elapsed.TotalMilliseconds);
    }

    static ImageRunResult ProcessOneImage(string file)
    {
        var swImage = Stopwatch.StartNew();
        using Bitmap original = new Bitmap(file);
        using Bitmap rgb = ConvertTo24Bit(original);
        float[] gray = ToGray(rgb);

        if (Method == 0 && BlurRadius > 0)
            gray = ApplyGaussianBlur(gray, rgb.Width, rgb.Height, BlurRadius);

        int[] labels = null;
        int tumorLabel = -1;
        var swProc = Stopwatch.StartNew();

        switch (Method)
        {
            case 0:
                labels = GraphSegmentation(gray, rgb.Width, rgb.Height);
                tumorLabel = BestTumorComponent(labels, gray, rgb.Width, rgb.Height);
                break;
            case 1:
                labels = GraphCutSegmentation(gray, rgb.Width, rgb.Height);
                tumorLabel = 1;
                break;
            case 2:
            case 3:
            case 4:
                (labels, tumorLabel) = GnnSegmentation(gray, rgb.Width, rgb.Height, Method);
                break;
        }
        swProc.Stop();

        using Bitmap result = CreateOverlay(rgb, labels, tumorLabel, Method);
        string outFile = Path.Combine(OutputFolderPath,
            Path.GetFileNameWithoutExtension(file) + "_" + GetMethodName(Method) + ".png");
        result.Save(outFile, ImageFormat.Png);
        swImage.Stop();

        return new ImageRunResult(outFile,
            new FullBenchmarkTiming(swProc.Elapsed.TotalMilliseconds, swImage.Elapsed.TotalMilliseconds));
    }

    static string GetMethodName(int m) => m switch
    {
        0 => "GraphSeg",
        1 => "GraphCut",
        2 => "GCN",
        3 => "HGNN",
        4 => "GAT",
        _ => "Unknown"
    };

    // ------------------------------------------------------------
    //  BENCHMARK SUPPORT
    // ------------------------------------------------------------
    class BenchmarkSummary
    {
        public double TotalProcessingTimeMs { get; private set; }
        public double TotalImageExecutionTimeMs { get; private set; }
        public double MinProcessingTimeMs { get; private set; } = double.MaxValue;
        public double MaxProcessingTimeMs { get; private set; }
        public double MinImageExecutionTimeMs { get; private set; } = double.MaxValue;
        public double MaxImageExecutionTimeMs { get; private set; }
        public int Count { get; private set; }
        public double AverageProcessingTimeMs => Count == 0 ? 0 : TotalProcessingTimeMs / Count;
        public double AverageImageExecutionTimeMs => Count == 0 ? 0 : TotalImageExecutionTimeMs / Count;

        public void Add(FullBenchmarkTiming timing)
        {
            Count++;
            TotalProcessingTimeMs += timing.ProcessingTimeMs;
            TotalImageExecutionTimeMs += timing.ImageExecutionTimeMs;
            MinProcessingTimeMs = Math.Min(MinProcessingTimeMs, timing.ProcessingTimeMs);
            MaxProcessingTimeMs = Math.Max(MaxProcessingTimeMs, timing.ProcessingTimeMs);
            MinImageExecutionTimeMs = Math.Min(MinImageExecutionTimeMs, timing.ImageExecutionTimeMs);
            MaxImageExecutionTimeMs = Math.Max(MaxImageExecutionTimeMs, timing.ImageExecutionTimeMs);
        }
    }

    record struct FullBenchmarkTiming(double ProcessingTimeMs, double ImageExecutionTimeMs);
    record struct ImageRunResult(string OutputPath, FullBenchmarkTiming Timing);

    static void PrintBenchmarkSummary(string label, BenchmarkSummary summary, double executionTimeMs)
    {
        Console.WriteLine($"\n{label} benchmark:");
        Console.WriteLine($"overall processing time: {summary.TotalProcessingTimeMs:F2} ms");
        Console.WriteLine($"average processing time: {summary.AverageProcessingTimeMs:F2} ms");
        Console.WriteLine($"min processing time: {summary.MinProcessingTimeMs:F2} ms");
        Console.WriteLine($"max processing time: {summary.MaxProcessingTimeMs:F2} ms");
        Console.WriteLine($"overall execution time: {executionTimeMs:F2} ms");
        Console.WriteLine($"average execution time: {summary.AverageImageExecutionTimeMs:F2} ms");
        Console.WriteLine($"min execution time: {summary.MinImageExecutionTimeMs:F2} ms");
        Console.WriteLine($"max execution time: {summary.MaxImageExecutionTimeMs:F2} ms");
    }

    public class GraphSegBenchmark
    {
        [Benchmark]
        public void FullApplicationExecution()
        {
            string assemblyPath = typeof(GraphSegBenchmark).Assembly.Location;
            ProcessStartInfo startInfo = new("dotnet")
            {
                ArgumentList = { assemblyPath, "--run-once" },
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true
            };

            using Process? process = Process.Start(startInfo);
            if (process is null) throw new InvalidOperationException("Failed to start benchmark process.");
            string output = process.StandardOutput.ReadToEnd();
            string error = process.StandardError.ReadToEnd();
            process.WaitForExit();

            if (process.ExitCode != 0)
                throw new InvalidOperationException($"Benchmark failed: {error}\n{output}");
        }
    }

    // ============================================================
    //  CPU GAUSSIAN BLUR
    // ============================================================
    static float[] ApplyGaussianBlur(float[] input, int width, int height, int radius)
    {
        if (radius <= 0) return input;

        // Build 1D Gaussian kernel
        float sigma = radius / 2f;
        int size = 2 * radius + 1;
        float[] kernel = new float[size];
        float sum = 0;
        for (int i = -radius; i <= radius; i++)
        {
            float v = MathF.Exp(-(i * i) / (2 * sigma * sigma));
            kernel[i + radius] = v;
            sum += v;
        }
        for (int i = 0; i < size; i++)
            kernel[i] /= sum;

        // Temporary buffer for horizontal pass
        float[] temp = new float[input.Length];

        // Horizontal pass
        for (int y = 0; y < height; y++)
        {
            int rowBase = y * width;
            for (int x = 0; x < width; x++)
            {
                float acc = 0;
                for (int k = -radius; k <= radius; k++)
                {
                    int xx = Math.Clamp(x + k, 0, width - 1);
                    acc += input[rowBase + xx] * kernel[k + radius];
                }
                temp[rowBase + x] = acc;
            }
        }

        // Vertical pass
        float[] output = new float[input.Length];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float acc = 0;
                for (int k = -radius; k <= radius; k++)
                {
                    int yy = Math.Clamp(y + k, 0, height - 1);
                    acc += temp[yy * width + x] * kernel[k + radius];
                }
                output[y * width + x] = acc;
            }
        }

        return output;
    }

    // ============================================================
    //  SEQUENTIAL GRAPH SEGMENTATION (Felzenszwalb & Huttenlocher)
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
    //  TUMOR COMPONENT SELECTION (for sequential graph only)
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
    //  GRAPH CUT SEGMENTATION (using custom Dinic max-flow)
    // ============================================================
    static int[] GraphCutSegmentation(float[] gray, int width, int height)
    {
        int n = width * height;
        int source = n;
        int sink = n + 1;
        int totalNodes = n + 2;

        Dinic dinic = new Dinic(totalNodes);

        float minGray = gray.Min();
        float maxGray = gray.Max();
        float range = Math.Max(maxGray - minGray, 1e-6f);

        for (int i = 0; i < n; i++)
        {
            float norm = (gray[i] - minGray) / range;
            float tumorCost = 1.0f - norm + 0.1f;
            float bgCost = norm + 0.1f;
            dinic.AddEdge(source, i, tumorCost, 0);
            dinic.AddEdge(i, sink, bgCost, 0);
        }

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int idx = y * width + x;
                if (x < width - 1)
                {
                    int nidx = idx + 1;
                    float diff = Math.Abs(gray[idx] - gray[nidx]);
                    float weight = Lambda * (float)Math.Exp(-diff / Sigma);
                    dinic.AddEdge(idx, nidx, weight, weight);
                }
                if (y < height - 1)
                {
                    int nidx = idx + width;
                    float diff = Math.Abs(gray[idx] - gray[nidx]);
                    float weight = Lambda * (float)Math.Exp(-diff / Sigma);
                    dinic.AddEdge(idx, nidx, weight, weight);
                }
            }
        }

        float flow = dinic.MaxFlow(source, sink);

        bool[] visited = new bool[totalNodes];
        dinic.DfsReach(source, visited);

        int[] labels = new int[n];
        for (int i = 0; i < n; i++)
            labels[i] = visited[i] ? 1 : 0;

        return labels;
    }

    // ------------------- Dinic Max-Flow Implementation -------------------
    class Dinic
    {
        class Edge
        {
            public int To;
            public float Cap;
            public int Rev;
        }

        List<Edge>[] graph;
        int[] level;
        int[] iter;

        public Dinic(int n)
        {
            graph = new List<Edge>[n];
            for (int i = 0; i < n; i++) graph[i] = new List<Edge>();
            level = new int[n];
            iter = new int[n];
        }

        public void AddEdge(int from, int to, float cap, float revCap)
        {
            Edge fwd = new Edge { To = to, Cap = cap, Rev = graph[to].Count };
            Edge rev = new Edge { To = from, Cap = revCap, Rev = graph[from].Count };
            graph[from].Add(fwd);
            graph[to].Add(rev);
        }

        void Bfs(int s)
        {
            for (int i = 0; i < level.Length; i++) level[i] = -1;
            Queue<int> q = new Queue<int>();
            level[s] = 0;
            q.Enqueue(s);
            while (q.Count > 0)
            {
                int v = q.Dequeue();
                foreach (var e in graph[v])
                {
                    if (e.Cap > 0 && level[e.To] < 0)
                    {
                        level[e.To] = level[v] + 1;
                        q.Enqueue(e.To);
                    }
                }
            }
        }

        float Dfs(int v, int t, float f)
        {
            if (v == t) return f;
            for (int i = iter[v]; i < graph[v].Count; i++)
            {
                iter[v] = i;
                Edge e = graph[v][i];
                if (e.Cap > 0 && level[v] < level[e.To])
                {
                    float d = Dfs(e.To, t, Math.Min(f, e.Cap));
                    if (d > 0)
                    {
                        e.Cap -= d;
                        graph[e.To][e.Rev].Cap += d;
                        return d;
                    }
                }
            }
            return 0;
        }

        public float MaxFlow(int s, int t)
        {
            float flow = 0;
            const float INF = 1e9f;
            while (true)
            {
                Bfs(s);
                if (level[t] < 0) break;
                for (int i = 0; i < iter.Length; i++) iter[i] = 0;
                float f;
                while ((f = Dfs(s, t, INF)) > 0)
                    flow += f;
            }
            return flow;
        }

        public void DfsReach(int v, bool[] visited)
        {
            visited[v] = true;
            foreach (var e in graph[v])
            {
                if (e.Cap > 0 && !visited[e.To])
                    DfsReach(e.To, visited);
            }
        }
    }

    // ============================================================
    //  GNN SEGMENTATION (GCN/HGNN/GAT) using ONNX Runtime
    // ============================================================
    static (int[] labels, int tumorLabel) GnnSegmentation(float[] gray, int width, int height, int method)
    {
        int n = width * height;
        int step = Math.Max(1, (int)Math.Sqrt(n / (double)SuperpixelCount));
        List<Center> centers = CreateCenters(gray, width, height, step);
        int[] superpixelLabels = RunSlic(gray, width, height, centers, step, SlicIterations, SlicCompactness);

        int numSuperpixels = centers.Count;
        var adjList = new List<int>[numSuperpixels];
        for (int i = 0; i < numSuperpixels; i++) adjList[i] = new List<int>();

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int idx = y * width + x;
                int sp = superpixelLabels[idx];
                if (x < width - 1)
                {
                    int sp2 = superpixelLabels[idx + 1];
                    if (sp != sp2 && !adjList[sp].Contains(sp2))
                        adjList[sp].Add(sp2);
                }
                if (y < height - 1)
                {
                    int sp2 = superpixelLabels[idx + width];
                    if (sp != sp2 && !adjList[sp].Contains(sp2))
                        adjList[sp].Add(sp2);
                }
            }
        }

        float[] sumGray = new float[numSuperpixels];
        float[] sumX = new float[numSuperpixels];
        float[] sumY = new float[numSuperpixels];
        int[] count = new int[numSuperpixels];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                int idx = y * width + x;
                int sp = superpixelLabels[idx];
                sumGray[sp] += gray[idx];
                sumX[sp] += x;
                sumY[sp] += y;
                count[sp]++;
            }

        float[][] features = new float[numSuperpixels][];
        for (int i = 0; i < numSuperpixels; i++)
        {
            if (count[i] == 0)
            {
                features[i] = new float[] { 0, 0, 0 };
                continue;
            }
            float meanGray = sumGray[i] / count[i];
            float meanX = sumX[i] / count[i];
            float meanY = sumY[i] / count[i];
            features[i] = new float[] { meanGray / 255f, meanX / width, meanY / height };
        }

        List<int> edgeRows = new List<int>();
        List<int> edgeCols = new List<int>();
        List<float> edgeVals = new List<float>();
        for (int i = 0; i < numSuperpixels; i++)
        {
            foreach (int j in adjList[i])
            {
                if (i < j)
                {
                    edgeRows.Add(i); edgeCols.Add(j); edgeVals.Add(1.0f);
                    edgeRows.Add(j); edgeCols.Add(i); edgeVals.Add(1.0f);
                }
            }
        }
        for (int i = 0; i < numSuperpixels; i++)
        {
            edgeRows.Add(i); edgeCols.Add(i); edgeVals.Add(1.0f);
        }

        string modelPath = method switch
        {
            2 => GcnModelPath,
            3 => HgnnModelPath,
            4 => GatModelPath,
            _ => throw new NotSupportedException()
        };

        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}");

        using var session = new InferenceSession(modelPath);
        var featureTensor = new DenseTensor<float>(features.SelectMany(f => f).ToArray(), new[] { numSuperpixels, 3 });
        var edgeIndexTensor = new DenseTensor<long>(edgeRows.Concat(edgeCols).Select(v => (long)v).ToArray(), new[] { 2, edgeRows.Count });
        var edgeWeightTensor = new DenseTensor<float>(edgeVals.ToArray(), new[] { edgeVals.Count });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("node_features", featureTensor),
            NamedOnnxValue.CreateFromTensor("edge_index", edgeIndexTensor),
            NamedOnnxValue.CreateFromTensor("edge_weight", edgeWeightTensor)
        };

        using var results = session.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();
        var probabilities = outputTensor.ToDenseTensor();

        int[] pixelLabels = new int[n];
        for (int i = 0; i < numSuperpixels; i++)
        {
            float tumorProb = probabilities[i, 1];
            foreach (int pixel in GetPixelsOfSuperpixel(superpixelLabels, i, width, height))
                pixelLabels[pixel] = tumorProb > 0.5f ? 1 : 0;
        }

        return (pixelLabels, 1);
    }

    static IEnumerable<int> GetPixelsOfSuperpixel(int[] labels, int sp, int width, int height)
    {
        for (int i = 0; i < labels.Length; i++)
            if (labels[i] == sp)
                yield return i;
    }

    // ---------- SLIC helpers ----------
    record struct Center(float Gray, float X, float Y);

    static List<Center> CreateCenters(float[] gray, int width, int height, int step)
    {
        List<Center> centers = new();
        for (int y = step / 2; y < height; y += step)
            for (int x = step / 2; x < width; x += step)
            {
                int idx = y * width + x;
                centers.Add(new Center(gray[idx], x, y));
            }
        return centers;
    }

    static int[] RunSlic(float[] gray, int width, int height, List<Center> centers, int step, int iterations, float compactness)
    {
        int n = width * height;
        int numCenters = centers.Count;
        float[] centerGray = new float[numCenters];
        float[] centerX = new float[numCenters];
        float[] centerY = new float[numCenters];
        int[] labels = new int[n];
        float spatialWeight = compactness * compactness / (step * step);
        int searchRadius = step * 2;

        for (int iter = 0; iter < iterations; iter++)
        {
            for (int i = 0; i < numCenters; i++)
            {
                centerGray[i] = centers[i].Gray;
                centerX[i] = centers[i].X;
                centerY[i] = centers[i].Y;
            }

            for (int pixel = 0; pixel < n; pixel++)
            {
                int x = pixel % width;
                int y = pixel / width;
                float bestDist = float.MaxValue;
                int bestLabel = 0;

                for (int c = 0; c < numCenters; c++)
                {
                    float dx = x - centerX[c];
                    float dy = y - centerY[c];
                    if (dx > searchRadius || dx < -searchRadius || dy > searchRadius || dy < -searchRadius)
                        continue;

                    float dg = gray[pixel] - centerGray[c];
                    float dist = dg * dg + spatialWeight * (dx * dx + dy * dy);
                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        bestLabel = c;
                    }
                }
                labels[pixel] = bestLabel;
            }

            float[] sumG = new float[numCenters];
            float[] sumX = new float[numCenters];
            float[] sumY = new float[numCenters];
            int[] count = new int[numCenters];

            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                {
                    int idx = y * width + x;
                    int c = labels[idx];
                    if (c < 0) continue;
                    sumG[c] += gray[idx];
                    sumX[c] += x;
                    sumY[c] += y;
                    count[c]++;
                }

            for (int i = 0; i < numCenters; i++)
                if (count[i] > 0)
                    centers[i] = new Center(sumG[i] / count[i], sumX[i] / count[i], sumY[i] / count[i]);
        }
        return labels;
    }

    // ============================================================
    //  OVERLAY GENERATION (supports binary labels for GraphCut/GNN)
    // ============================================================
    static unsafe Bitmap CreateOverlay(Bitmap original, int[] labels, int tumorLabel, int method)
    {
        int w = original.Width, h = original.Height;
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
                bool isTumor = (method == 0) ? labels[idx] == tumorLabel : labels[idx] == 1;

                if (isTumor)
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

    // ============================================================
    //  IMAGE UTILITIES
    // ============================================================
    static bool IsSupportedImage(string path)
    {
        string ext = Path.GetExtension(path).ToLowerInvariant();
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
}