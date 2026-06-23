using System.Drawing;
using System.Drawing.Imaging;
using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;

// SLIC and tumor-candidate settings are chosen automatically per image below.
const string InputFolderPath = @"E:\TPC Preprocessing\Preprocess Dataset"; // Put the folder containing preprocessed grayscale images here.
const string OutputFolderPath = @"E:\TPC_SLIC\Parallel Slic (Full Benchmark Check)\SLIC_GPU\output";
const bool RunBenchmarkDotNet = false; // Set true and click Run to execute BenchmarkDotNet instead of normal output generation.

bool runOnce = args.Contains("--run-once", StringComparer.OrdinalIgnoreCase);
if ((RunBenchmarkDotNet && !runOnce) || args.Contains("--benchmark", StringComparer.OrdinalIgnoreCase))
{
    BenchmarkRunner.Run<GpuSlicBenchmark>();
    return;
}

Stopwatch executionStopwatch = Stopwatch.StartNew();
string inputFolder = ResolveInputFolder(InputFolderPath);
string outputDir = PrepareOutputFolder(OutputFolderPath, inputFolder);
string[] inputPaths = FindInputImages(inputFolder);

ImageData warmupImage = LoadPreprocessedImage(inputPaths[0]);
RunGpuSlic(warmupImage); // Warm-up run for ILGPU kernel compilation. Do not include this in timing.
BenchmarkSummary benchmark = new();

foreach (string inputPath in inputPaths)
{
    ImageRunResult result = ProcessImage(inputFolder, outputDir, inputPath, RunGpuSlic);
    benchmark.Add(result.Timing);
    Console.WriteLine("Output: " + result.OutputPath);
}

executionStopwatch.Stop();
PrintBenchmark("GPU", benchmark, executionStopwatch.Elapsed.TotalMilliseconds);

static string ResolveInputFolder(string inputFolderPath)
{
    if (string.IsNullOrWhiteSpace(inputFolderPath))
        throw new ArgumentException("Please set InputFolderPath in Program.cs.");

    string inputFolder = Path.GetFullPath(inputFolderPath);
    if (!Directory.Exists(inputFolder))
        throw new DirectoryNotFoundException("Input folder was not found: " + inputFolder);

    return inputFolder;
}

static string PrepareOutputFolder(string outputFolderPath, string inputFolder)
{
    string outputDir = string.IsNullOrWhiteSpace(outputFolderPath)
        ? inputFolder
        : Path.GetFullPath(outputFolderPath);
    Directory.CreateDirectory(outputDir);
    return outputDir;
}

static string[] FindInputImages(string inputFolder)
{
    string[] inputPaths = Directory
        .EnumerateFiles(inputFolder, "*.*", SearchOption.AllDirectories)
        .Where(IsSupportedImage)
        .OrderBy(path => path)
        .ToArray();

    if (inputPaths.Length == 0)
        throw new ArgumentException("No input images were found in: " + inputFolder);

    return inputPaths;
}

static ImageRunResult ProcessImage(
    string inputFolder,
    string outputDir,
    string inputPath,
    Func<ImageData, int[]> runSlic)
{
    Stopwatch imageExecutionStopwatch = Stopwatch.StartNew();
    ImageData image = LoadAndReportImage(inputPath);

    Stopwatch processingStopwatch = Stopwatch.StartNew();
    int[] labels = runSlic(image);
    bool[] tumorMask = DetectTumorCandidates(image, labels);
    processingStopwatch.Stop();

    string outputPath = SaveTumorCandidateOutput(inputFolder, outputDir, inputPath, image, tumorMask);
    imageExecutionStopwatch.Stop();

    return new ImageRunResult(
        outputPath,
        new FullBenchmarkTiming(
            processingStopwatch.Elapsed.TotalMilliseconds,
            imageExecutionStopwatch.Elapsed.TotalMilliseconds));
}

static ImageData LoadAndReportImage(string inputPath)
{
    ImageData image = LoadPreprocessedImage(inputPath);
    Console.WriteLine("Preprocessed image: " + inputPath);
    Console.WriteLine("Size: " + image.Width + " x " + image.Height);
    return image;
}

static string SaveTumorCandidateOutput(string inputFolder, string outputDir, string inputPath, ImageData image, bool[] tumorMask)
{
    string relativeFolder = Path.GetDirectoryName(Path.GetRelativePath(inputFolder, inputPath)) ?? "";
    string outputSubFolder = Path.Combine(outputDir, relativeFolder);
    Directory.CreateDirectory(outputSubFolder);

    string outputPath = Path.Combine(
        outputSubFolder,
        Path.GetFileNameWithoutExtension(inputPath) + "_tumor_candidate.jpg");

    SaveTumorOverlay(outputPath, image, tumorMask);
    return outputPath;
}

static void PrintBenchmark(string label, BenchmarkSummary benchmark, double executionTimeMs)
{
    Console.WriteLine($"{label} benchmark:");
    Console.WriteLine($"{label} overall processing time: {benchmark.TotalProcessingTimeMs:F2} ms");
    Console.WriteLine($"{label} average processing time: {benchmark.AverageProcessingTimeMs:F2} ms");
    Console.WriteLine($"{label} min processing time: {benchmark.MinProcessingTimeMs:F2} ms");
    Console.WriteLine($"{label} max processing time: {benchmark.MaxProcessingTimeMs:F2} ms");
    Console.WriteLine($"{label} overall execution time: {executionTimeMs:F2} ms");
    Console.WriteLine($"{label} average execution time: {benchmark.AverageImageExecutionTimeMs:F2} ms");
    Console.WriteLine($"{label} min execution time: {benchmark.MinImageExecutionTimeMs:F2} ms");
    Console.WriteLine($"{label} max execution time: {benchmark.MaxImageExecutionTimeMs:F2} ms");
}

static bool IsSupportedImage(string path)
{
    string ext = Path.GetExtension(path).ToLowerInvariant();
    return ext is ".jpg" or ".jpeg" or ".png" or ".bmp";
}

static ImageData LoadPreprocessedImage(string path)
{
    using Bitmap source = new Bitmap(path);
    using Bitmap bitmap = ConvertTo24BitRgb(source);

    int w = bitmap.Width, h = bitmap.Height, n = w * h;
    byte[] r = new byte[n], g = new byte[n], b = new byte[n];
    float[] gray = new float[n];

    BitmapData data = bitmap.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
    unsafe
    {
        byte* ptr = (byte*)data.Scan0;
        for (int y = 0; y < h; y++)
        {
            byte* row = ptr + y * data.Stride;
            for (int x = 0; x < w; x++)
            {
                int i = y * w + x, p = x * 3;
                byte grayValue = row[p]; // Preprocessing has already made R, G, and B equal.
                b[i] = grayValue;
                g[i] = grayValue;
                r[i] = grayValue;
                gray[i] = grayValue;
            }
        }
    }
    bitmap.UnlockBits(data);
    return new ImageData(w, h, r, g, b, gray);
}

static Bitmap ConvertTo24BitRgb(Bitmap source)
{
    Bitmap bitmap = new Bitmap(source.Width, source.Height, PixelFormat.Format24bppRgb);
    using Graphics graphics = Graphics.FromImage(bitmap);
    graphics.DrawImage(source, new Rectangle(0, 0, source.Width, source.Height));
    return bitmap;
}

static int[] RunGpuSlic(ImageData image)
{
    int n = image.Width * image.Height;
    AutoSlicSettings settings = ChooseSlicSettings(image);
    int step = settings.Step;
    List<Center> centers = CreateCenters(image, step);
    int[] labels = new int[n];

    using Context context = Context.CreateDefault();
    Device device = PickGpuDevice(context);
    using Accelerator accelerator = device.CreateAccelerator(context);
    Console.WriteLine("ILGPU accelerator: " + accelerator);
    PrintSlicSettings(settings);

    using MemoryBuffer1D<float, Stride1D.Dense> dGray = accelerator.Allocate1D(image.Gray);
    using MemoryBuffer1D<float, Stride1D.Dense> dCg = accelerator.Allocate1D<float>(centers.Count);
    using MemoryBuffer1D<float, Stride1D.Dense> dCx = accelerator.Allocate1D<float>(centers.Count);
    using MemoryBuffer1D<float, Stride1D.Dense> dCy = accelerator.Allocate1D<float>(centers.Count);
    using MemoryBuffer1D<int, Stride1D.Dense> dLabels = accelerator.Allocate1D<int>(n);

    var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, SlicViews, SlicSettings>(AssignLabelsKernel);
    float[] cg = new float[centers.Count], cx = new float[centers.Count], cy = new float[centers.Count];

    for (int iter = 0; iter < settings.Iterations; iter++)
    {
        for (int i = 0; i < centers.Count; i++)
        {
            cg[i] = centers[i].Gray;
            cx[i] = centers[i].X;
            cy[i] = centers[i].Y;
        }

        dCg.CopyFromCPU(cg);
        dCx.CopyFromCPU(cx);
        dCy.CopyFromCPU(cy);

        kernel(n, new SlicViews(dGray.View, dCg.View, dCx.View, dCy.View, dLabels.View),
            new SlicSettings(image.Width, centers.Count, settings.SearchRadius, settings.SpatialWeight));
        accelerator.Synchronize();
        dLabels.CopyToCPU(labels);
        UpdateCenters(image, labels, centers);
    }

    return labels;
}

static Device PickGpuDevice(Context context)
{
    var cuda = context.GetCudaDevices();
    if (cuda.Count > 0) return cuda[0];

    var openCl = context.GetCLDevices();
    if (openCl.Count > 0) return openCl[0];

    throw new InvalidOperationException("No CUDA/OpenCL GPU device found. Please install NVIDIA CUDA driver or OpenCL runtime.");
}

static AutoSlicSettings ChooseSlicSettings(ImageData image)
{
    int n = image.Width * image.Height;
    int minDimension = Math.Max(1, Math.Min(image.Width, image.Height));
    float tissueThreshold = Otsu(image.Gray) * 0.6f;
    int tissuePixels = 0;

    foreach (float gray in image.Gray)
        if (gray > tissueThreshold)
            tissuePixels++;

    float tissueRatio = tissuePixels / (float)Math.Max(1, n);
    int desiredStep = Math.Clamp((int)MathF.Round(minDimension / 24f), 8, 28);
    if (tissueRatio > 0 && tissueRatio < 0.35f)
        desiredStep = Math.Max(6, desiredStep - 2);

    int superpixels = Math.Clamp((int)MathF.Round(n / (float)(desiredStep * desiredStep)), 80, 1400);
    int step = Math.Max(1, (int)Math.Sqrt(n / (double)superpixels));
    float contrast = EstimateRobustContrast(image.Gray);
    float compactnessRatio = contrast < 45f ? 0.58f : contrast > 110f ? 0.38f : 0.48f;
    float compactness = Math.Clamp(step * compactnessRatio, 6f, 16f);
    int iterations = Math.Clamp(5 + (int)MathF.Round(MathF.Log2(MathF.Max(2, step))), 6, 10);

    return new AutoSlicSettings(superpixels, step, iterations, compactness);
}

static float EstimateRobustContrast(float[] gray)
{
    List<float> sorted = new(gray);
    sorted.Sort();
    return Percentile(sorted, 90f) - Percentile(sorted, 10f);
}

static void PrintSlicSettings(AutoSlicSettings settings)
{
    Console.WriteLine(
        $"Auto SLIC settings: superpixels={settings.Superpixels}, step={settings.Step}, compactness={settings.Compactness:F1}, iterations={settings.Iterations}");
}

static List<Center> CreateCenters(ImageData image, int step)
{
    List<Center> centers = new();
    for (int y = step / 2; y < image.Height; y += step)
        for (int x = step / 2; x < image.Width; x += step)
        {
            int i = y * image.Width + x;
            centers.Add(new Center(image.Gray[i], x, y));
        }
    return centers;
}

static void UpdateCenters(ImageData image, int[] labels, List<Center> centers)
{
    float[] sumG = new float[centers.Count], sumX = new float[centers.Count], sumY = new float[centers.Count];
    int[] count = new int[centers.Count];

    for (int y = 0; y < image.Height; y++)
        for (int x = 0; x < image.Width; x++)
        {
            int i = y * image.Width + x, label = labels[i];
            if (label < 0) continue;
            sumG[label] += image.Gray[i];
            sumX[label] += x;
            sumY[label] += y;
            count[label]++;
        }

    for (int i = 0; i < centers.Count; i++)
    {
        if (count[i] == 0) continue;
        centers[i] = new Center(sumG[i] / count[i], sumX[i] / count[i], sumY[i] / count[i]);
    }
}

static bool[] DetectTumorCandidates(ImageData image, int[] labels)
{
    int labelCount = labels.Max() + 1;
    LabelStats[] stats = BuildLabelStats(image, labels, labelCount, out BrainInfo brain);
    bool[] candidateLabels = PickBrightCandidateLabels(stats, brain);
    bool[] mask = new bool[labels.Length];

    for (int i = 0; i < labels.Length; i++)
    {
        int x = i % image.Width, y = i / image.Width;
        mask[i] = candidateLabels[labels[i]] &&
                  image.Gray[i] > brain.TissueThreshold * 0.8f &&
                  InsideBrainRoi(x, y, brain);
    }

    CleanupCandidateRegions(mask, image.Width, image.Height, brain);
    return mask;
}

static LabelStats[] BuildLabelStats(ImageData image, int[] labels, int labelCount, out BrainInfo brain)
{
    brain = BuildBrainInfo(image);
    float tissue = brain.TissueThreshold;
    float bx = brain.X, by = brain.Y, radius = brain.Radius;

    LabelStats[] stats = Enumerable.Range(0, labelCount).Select(_ => new LabelStats()).ToArray();
    int margin = Math.Max(2, Math.Min(image.Width, image.Height) / 100);

    for (int y = 0; y < image.Height; y++)
        for (int x = 0; x < image.Width; x++)
        {
            int i = y * image.Width + x;
            LabelStats s = stats[labels[i]];
            float gray = image.Gray[i];
            s.Sum += gray;
            s.SumSq += gray * gray;
            s.SumX += x;
            s.SumY += y;
            s.Count++;
            if (gray > tissue) s.Tissue++;
            if (x < margin || y < margin || x >= image.Width - margin || y >= image.Height - margin) s.Border++;
            if (TouchesBackground(image, x, y, tissue)) s.BackgroundNeighbor++;
            if (!InsideBrainRoi(x, y, brain)) s.OutsideRoi++;
            float dx = x - bx, dy = y - by;
            if (MathF.Sqrt(dx * dx + dy * dy) > radius) s.OutsideRadius++;
        }

    return stats;
}

static BrainInfo BuildBrainInfo(ImageData image)
{
    float tissue = Otsu(image.Gray) * 0.6f;
    List<float> xs = new();
    List<float> ys = new();
    double sx = 0, sy = 0;

    for (int y = 0; y < image.Height; y++)
        for (int x = 0; x < image.Width; x++)
        {
            int i = y * image.Width + x;
            if (image.Gray[i] <= tissue) continue;
            xs.Add(x);
            ys.Add(y);
            sx += x;
            sy += y;
        }

    if (xs.Count == 0)
    {
        float fallbackRadius = Math.Min(image.Width, image.Height) * 0.5f;
        return new BrainInfo(image.Width / 2f, image.Height / 2f, fallbackRadius, tissue, 0, 0, image.Width - 1, image.Height - 1);
    }

    xs.Sort();
    ys.Sort();
    float bx = (float)(sx / xs.Count);
    float by = (float)(sy / ys.Count);
    int roiLeft = Math.Clamp((int)MathF.Floor(Percentile(xs, 3f)), 0, image.Width - 1);
    int roiTop = Math.Clamp((int)MathF.Floor(Percentile(ys, 3f)), 0, image.Height - 1);
    int roiRight = Math.Clamp((int)MathF.Ceiling(Percentile(xs, 97f)), roiLeft, image.Width - 1);
    int roiBottom = Math.Clamp((int)MathF.Ceiling(Percentile(ys, 97f)), roiTop, image.Height - 1);
    List<float> distances = new(xs.Count);

    for (int y = 0; y < image.Height; y++)
        for (int x = 0; x < image.Width; x++)
        {
            int i = y * image.Width + x;
            if (image.Gray[i] <= tissue) continue;
            float dx = x - bx, dy = y - by;
            distances.Add(MathF.Sqrt(dx * dx + dy * dy));
        }

    distances.Sort();
    float minDimension = Math.Min(image.Width, image.Height);
    float radius = Math.Clamp(Percentile(distances, 90f), minDimension * 0.20f, minDimension * 0.55f);

    return new BrainInfo(bx, by, radius, tissue, roiLeft, roiTop, roiRight, roiBottom);
}

static bool[] PickBrightCandidateLabels(LabelStats[] stats, BrainInfo brain)
{
    List<float> means = new();
    double sum = 0, sumSq = 0;
    int count = 0;

    foreach (LabelStats s in stats)
    {
        if (!IsBrainLabel(s, brain)) continue;
        means.Add(s.Mean);
        sum += s.Sum;
        sumSq += s.SumSq;
        count += s.Count;
    }

    bool[] candidates = new bool[stats.Length];
    if (means.Count == 0)
        return candidates;

    means.Sort();
    float mean = count == 0 ? 0 : (float)(sum / count);
    float std = count == 0 ? 0 : MathF.Sqrt(MathF.Max(0, (float)(sumSq / count - mean * mean)));
    float q25 = Percentile(means, 25f);
    float q75 = Percentile(means, 75f);
    float q90 = Percentile(means, 90f);
    float spread = MathF.Max(1f, q90 - q25);
    float sensitivity = Math.Clamp(spread / 96f, 0.45f, 0.85f);
    float threshold = MathF.Min(q90, MathF.Max(q75, mean + sensitivity * std));

    for (int i = 0; i < stats.Length; i++)
        candidates[i] = IsBrainLabel(stats[i], brain) && stats[i].Mean >= threshold;

    return candidates;
}

static bool IsBrainLabel(LabelStats s, BrainInfo brain)
{
    if (s.Count == 0) return false;
    float dx = s.CenterX - brain.X, dy = s.CenterY - brain.Y;
    return s.TissueRatio >= 0.50f &&
           s.BorderRatio <= 0.30f &&
           s.BackgroundRatio <= 0.35f &&
           s.OutsideRoiRatio <= 0.70f &&
           s.OutsideRatio <= 0.10f &&
           InsideBrainRoi((int)s.CenterX, (int)s.CenterY, brain) &&
           MathF.Sqrt(dx * dx + dy * dy) <= brain.Radius;
}

static void CleanupCandidateRegions(bool[] mask, int w, int h, BrainInfo brain)
{
    bool[] seen = new bool[mask.Length];
    int[] q = new int[mask.Length];
    List<int> region = new();
    int minPixels = Math.Max(32, mask.Length / 3000);

    for (int start = 0; start < mask.Length; start++)
    {
        if (!mask[start] || seen[start]) continue;
        region.Clear();
        int head = 0, tail = 0, minX = w, minY = h, maxX = 0, maxY = 0;
        float distanceSum = 0, sumX = 0, sumY = 0;
        int outsideRoiCount = 0;
        q[tail++] = start;
        seen[start] = true;

        while (head < tail)
        {
            int p = q[head++], x = p % w, y = p / w;
            region.Add(p);
            minX = Math.Min(minX, x); maxX = Math.Max(maxX, x);
            minY = Math.Min(minY, y); maxY = Math.Max(maxY, y);
            sumX += x; sumY += y;
            if (!InsideBrainRoi(x, y, brain)) outsideRoiCount++;
            float dx = x - brain.X, dy = y - brain.Y;
            distanceSum += MathF.Sqrt(dx * dx + dy * dy);
            Add(p - 1, x > 0); Add(p + 1, x < w - 1); Add(p - w, y > 0); Add(p + w, y < h - 1);
        }

        int boxW = maxX - minX + 1, boxH = maxY - minY + 1;
        float aspect = Math.Max(boxW, boxH) / (float)Math.Max(1, Math.Min(boxW, boxH));
        float density = region.Count / (float)Math.Max(1, boxW * boxH);
        float meanDistance = distanceSum / region.Count;
        float centerX = sumX / region.Count, centerY = sumY / region.Count;
        bool remove = region.Count < minPixels ||
                      !InsideBrainRoi((int)centerX, (int)centerY, brain) ||
                      outsideRoiCount > region.Count * 0.05f ||
                      meanDistance > brain.Radius * 0.70f ||
                      (aspect > 3f && density < 0.80f) ||
                      (meanDistance > brain.Radius * 0.82f && (aspect > 2f || density < 0.65f));

        if (remove) foreach (int p in region) mask[p] = false;

        void Add(int p, bool inside)
        {
            if (!inside || seen[p] || !mask[p]) return;
            seen[p] = true;
            q[tail++] = p;
        }
    }
}

static bool InsideBrainRoi(int x, int y, BrainInfo brain)
{
    return x >= brain.RoiLeft && x <= brain.RoiRight && y >= brain.RoiTop && y <= brain.RoiBottom;
}

static void SaveTumorOverlay(string path, ImageData image, bool[] mask)
{
    using Bitmap output = new Bitmap(image.Width, image.Height, PixelFormat.Format24bppRgb);
    BitmapData data = output.LockBits(new Rectangle(0, 0, image.Width, image.Height), ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);
    unsafe
    {
        byte* ptr = (byte*)data.Scan0;
        for (int y = 0; y < image.Height; y++)
        {
            byte* row = ptr + y * data.Stride;
            for (int x = 0; x < image.Width; x++)
            {
                int i = y * image.Width + x, p = x * 3;
                byte r = image.R[i], g = image.G[i], b = image.B[i];
                if (mask[i]) { r = Blend(r, 255); g = Blend(g, 255); b = Blend(b, 0); }
                if (IsMaskBoundary(mask, image.Width, image.Height, x, y)) { r = 0; g = 255; b = 0; }
                row[p] = b; row[p + 1] = g; row[p + 2] = r;
            }
        }
    }
    output.UnlockBits(data);
    output.Save(path, ImageFormat.Jpeg);
}

static bool TouchesBackground(ImageData img, int x, int y, float th)
{
    int w = img.Width, h = img.Height, i = y * w + x;
    return x == 0 || y == 0 || x == w - 1 || y == h - 1 ||
           img.Gray[i - 1] <= th || img.Gray[i + 1] <= th || img.Gray[i - w] <= th || img.Gray[i + w] <= th;
}

static bool IsMaskBoundary(bool[] m, int w, int h, int x, int y)
{
    int i = y * w + x;
    return m[i] && ((x > 0 && !m[i - 1]) || (x < w - 1 && !m[i + 1]) || (y > 0 && !m[i - w]) || (y < h - 1 && !m[i + w]));
}

static float Otsu(float[] gray)
{
    int[] hist = new int[256];
    foreach (float v in gray) hist[(int)Math.Clamp(v, 0, 255)]++;
    float total = gray.Length, sum = 0, sumB = 0, wB = 0, best = 0;
    int threshold = 0;
    for (int i = 0; i < 256; i++) sum += i * hist[i];
    for (int i = 0; i < 256; i++)
    {
        wB += hist[i];
        if (wB == 0) continue;
        float wF = total - wB;
        if (wF == 0) break;
        sumB += i * hist[i];
        float diff = sumB / wB - (sum - sumB) / wF;
        float between = wB * wF * diff * diff;
        if (between > best) { best = between; threshold = i; }
    }
    return threshold;
}

static float Percentile(List<float> sorted, float pct)
{
    if (sorted.Count == 0) return float.MaxValue;
    float pos = pct / 100f * (sorted.Count - 1);
    int lo = (int)MathF.Floor(pos), hi = (int)MathF.Ceiling(pos);
    return lo == hi ? sorted[lo] : sorted[lo] * (hi - pos) + sorted[hi] * (pos - lo);
}

static byte Blend(byte original, byte overlay) => (byte)(original * 0.55f + overlay * 0.45f);

static void AssignLabelsKernel(Index1D pixel, SlicViews v, SlicSettings s)
{
    int i = pixel, x = i % s.Width, y = i / s.Width;
    float best = float.MaxValue;
    int bestLabel = 0;

    for (int c = 0; c < s.CenterCount; c++)
    {
        float dx = x - v.CenterX[c], dy = y - v.CenterY[c];
        if (dx > s.SearchRadius || dx < -s.SearchRadius || dy > s.SearchRadius || dy < -s.SearchRadius) continue;
        float dg = v.Gray[i] - v.CenterGray[c];
        float d = dg * dg + s.SpatialWeight * (dx * dx + dy * dy);
        if (d < best) { best = d; bestLabel = c; }
    }

    v.Labels[i] = bestLabel;
}

public class GpuSlicBenchmark
{
    [Benchmark]
    public void FullApplicationExecution()
    {
        BenchmarkProcessRunner.RunCurrentAssembly();
    }
}

static class BenchmarkProcessRunner
{
    public static void RunCurrentAssembly()
    {
        string assemblyPath = typeof(BenchmarkProcessRunner).Assembly.Location;
        ProcessStartInfo startInfo = new("dotnet");
        startInfo.ArgumentList.Add(assemblyPath);
        startInfo.ArgumentList.Add("--run-once");
        startInfo.UseShellExecute = false;
        startInfo.RedirectStandardOutput = true;
        startInfo.RedirectStandardError = true;

        using Process? process = Process.Start(startInfo);
        if (process is null)
            throw new InvalidOperationException("Failed to start benchmark process.");

        string output = process.StandardOutput.ReadToEnd();
        string error = process.StandardError.ReadToEnd();
        process.WaitForExit();

        if (process.ExitCode != 0)
            throw new InvalidOperationException("Benchmark process failed: " + error + output);
    }
}

record struct FullBenchmarkTiming(double ProcessingTimeMs, double ImageExecutionTimeMs);
record struct ImageRunResult(string OutputPath, FullBenchmarkTiming Timing);

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

record ImageData(int Width, int Height, byte[] R, byte[] G, byte[] B, float[] Gray);
record struct AutoSlicSettings(int Superpixels, int Step, int Iterations, float Compactness)
{
    public int SearchRadius => Step * 2;
    public float SpatialWeight => Compactness * Compactness / Math.Max(1, Step * Step);
}
record struct Center(float Gray, float X, float Y);
record struct BrainInfo(
    float X,
    float Y,
    float Radius,
    float TissueThreshold,
    int RoiLeft,
    int RoiTop,
    int RoiRight,
    int RoiBottom);

class LabelStats
{
    public float Sum, SumSq, SumX, SumY;
    public int Count, Tissue, Border, BackgroundNeighbor, OutsideRadius, OutsideRoi;
    public float Mean => Count == 0 ? 0 : Sum / Count;
    public float CenterX => Count == 0 ? 0 : SumX / Count;
    public float CenterY => Count == 0 ? 0 : SumY / Count;
    public float TissueRatio => Count == 0 ? 0 : Tissue / (float)Count;
    public float BorderRatio => Count == 0 ? 0 : Border / (float)Count;
    public float BackgroundRatio => Count == 0 ? 0 : BackgroundNeighbor / (float)Count;
    public float OutsideRatio => Count == 0 ? 0 : OutsideRadius / (float)Count;
    public float OutsideRoiRatio => Count == 0 ? 0 : OutsideRoi / (float)Count;
}

public readonly record struct SlicViews(
    ArrayView<float> Gray,
    ArrayView<float> CenterGray,
    ArrayView<float> CenterX,
    ArrayView<float> CenterY,
    ArrayView<int> Labels);

public readonly record struct SlicSettings(int Width, int CenterCount, int SearchRadius, float SpatialWeight);
