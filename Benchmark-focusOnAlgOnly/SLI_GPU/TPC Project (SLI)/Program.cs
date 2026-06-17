using System.Drawing;
using System.Drawing.Imaging;
using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using Microsoft.ConcurrencyVisualizer.Instrumentation;

const int Superpixels = 500; // Number of superpixels. Higher = smaller regions, more detail.
const int Iterations = 8; // SLIC update rounds. Higher = more stable, but slower.
const float Compactness = 10f; // Shape control. Higher = smoother/squarer superpixels, lower = follows edges more.
const float TumorSensitivity = 0.7f; // Tumor threshold strictness. Higher = fewer highlighted areas, lower = more sensitive.
const float TumorPercentile = 75f; // Keeps only bright superpixels above this percentile. Higher = fewer candidates.
const float BrainRadiusFactor = 0.30f; // Removes far outer regions. Lower = stricter against skull/skin edge.
const bool UseBrainRoi = true; // true = only search inside the brain ROI below.
const float BrainRoiLeft = 0.56f; // ROI left boundary as image width ratio. Lower if tumor is missed on the left.
const float BrainRoiTop = 0.41f; // ROI top boundary as image height ratio. Increase to ignore top skull edge.
const float BrainRoiRight = 0.65f; // ROI right boundary as image width ratio. Lower to ignore right skull edge.
const float BrainRoiBottom = 0.49f; // ROI bottom boundary as image height ratio. Lower to ignore face/neck areas.
const string InputFolderPath = @"E:\TPC Preprocessing\Preprocess Dataset"; // Put the folder containing preprocessed grayscale images here.
const string OutputFolderPath = @"E:\TPC Project (SLI)\output";
const bool RunBenchmarkDotNet = false; // Set true and click Run to execute BenchmarkDotNet instead of normal output generation.

bool runOnce = args.Contains("--run-once", StringComparer.OrdinalIgnoreCase);
bool runSlicOnly = args.Contains("--slic-only", StringComparer.OrdinalIgnoreCase);
if ((RunBenchmarkDotNet && !runOnce) || args.Contains("--benchmark", StringComparer.OrdinalIgnoreCase))
{
    BenchmarkRunner.Run<GpuSlicBenchmark>();
    return;
}

WriteProfileFlag("GPU SLIC evaluation start");

if (string.IsNullOrWhiteSpace(InputFolderPath))
    throw new ArgumentException("Please set InputFolderPath in Program.cs.");

string inputFolder = Path.GetFullPath(InputFolderPath);
if (!Directory.Exists(inputFolder))
    throw new DirectoryNotFoundException("Input folder was not found: " + inputFolder);

string outputDir = string.IsNullOrWhiteSpace(OutputFolderPath)
    ? inputFolder
    : Path.GetFullPath(OutputFolderPath);
Directory.CreateDirectory(outputDir);

string[] inputPaths = Directory
    .EnumerateFiles(inputFolder, "*.*", SearchOption.AllDirectories)
    .Where(IsSupportedImage)
    .OrderBy(path => path)
    .ToArray();

if (inputPaths.Length == 0)
    throw new ArgumentException("No input images were found in: " + inputFolder);

using (EnterProfileSpan("GPU warm-up"))
{
    ImageData warmupImage = LoadPreprocessedImage(inputPaths[0]);
    RunGpuSlic(warmupImage); // Warm-up run for ILGPU kernel compilation. Do not include this in timing.
}

double totalProcessingTimeMs = 0;
double totalExecutionTimeMs = 0;
double minProcessingTimeMs = double.MaxValue;
double maxProcessingTimeMs = 0;
double minExecutionTimeMs = double.MaxValue;
double maxExecutionTimeMs = 0;

foreach (string inputPath in inputPaths)
{
    string imageName = Path.GetFileName(inputPath);
    ImageData image;
    int[] labels;
    bool[] tumorMask;
    string outputPath;

    image = LoadPreprocessedImage(inputPath);

    Console.WriteLine("Preprocessed image: " + inputPath);
    Console.WriteLine("Size: " + image.Width + " x " + image.Height);

    WriteProfileFlag("GPU SLIC start: " + imageName);
    Stopwatch executionStopwatch = Stopwatch.StartNew();
    using (EnterProfileSpan("GPU SLIC execution: " + imageName))
    {
        Stopwatch processingStopwatch = Stopwatch.StartNew();
        using (EnterProfileSpan("GPU SLIC processing: " + imageName))
        {
            labels = RunGpuSlic(image);
        }
        processingStopwatch.Stop();

        double processingTimeMs = processingStopwatch.Elapsed.TotalMilliseconds;
        totalProcessingTimeMs += processingTimeMs;
        minProcessingTimeMs = Math.Min(minProcessingTimeMs, processingTimeMs);
        maxProcessingTimeMs = Math.Max(maxProcessingTimeMs, processingTimeMs);
    }
    executionStopwatch.Stop();
    WriteProfileFlag("GPU SLIC end: " + imageName);

    double executionTimeMs = executionStopwatch.Elapsed.TotalMilliseconds;
    totalExecutionTimeMs += executionTimeMs;
    minExecutionTimeMs = Math.Min(minExecutionTimeMs, executionTimeMs);
    maxExecutionTimeMs = Math.Max(maxExecutionTimeMs, executionTimeMs);

    if (runSlicOnly)
        continue;

    tumorMask = DetectTumorCandidates(image, labels);

    string relativeFolder = Path.GetDirectoryName(Path.GetRelativePath(inputFolder, inputPath)) ?? "";
    string outputSubFolder = Path.Combine(outputDir, relativeFolder);
    Directory.CreateDirectory(outputSubFolder);

    outputPath = Path.Combine(
        outputSubFolder,
        Path.GetFileNameWithoutExtension(inputPath) + "_tumor_candidate.jpg");

    SaveTumorOverlay(outputPath, image, tumorMask);

    Console.WriteLine("Output: " + outputPath);
}

WriteProfileFlag("GPU SLIC evaluation end");

PrintBenchmark(
    "GPU",
    totalProcessingTimeMs,
    totalExecutionTimeMs,
    minProcessingTimeMs,
    maxProcessingTimeMs,
    minExecutionTimeMs,
    maxExecutionTimeMs,
    inputPaths.Length);

static void PrintBenchmark(
    string label,
    double processingTimeMs,
    double executionTimeMs,
    double minProcessingTimeMs,
    double maxProcessingTimeMs,
    double minExecutionTimeMs,
    double maxExecutionTimeMs,
    int imageCount)
{
    double averageProcessingTimeMs = processingTimeMs / imageCount;
    double averageExecutionTimeMs = executionTimeMs / imageCount;

    Console.WriteLine($"{label} SLIC benchmark:");
    Console.WriteLine($"{label} SLIC overall processing time: {processingTimeMs:F2} ms");
    Console.WriteLine($"{label} SLIC average processing time: {averageProcessingTimeMs:F2} ms");
    Console.WriteLine($"{label} SLIC min processing time: {minProcessingTimeMs:F2} ms");
    Console.WriteLine($"{label} SLIC max processing time: {maxProcessingTimeMs:F2} ms");
    Console.WriteLine($"{label} SLIC overall execution time: {executionTimeMs:F2} ms");
    Console.WriteLine($"{label} SLIC average execution time: {averageExecutionTimeMs:F2} ms");
    Console.WriteLine($"{label} SLIC min execution time: {minExecutionTimeMs:F2} ms");
    Console.WriteLine($"{label} SLIC max execution time: {maxExecutionTimeMs:F2} ms");
}

static IDisposable EnterProfileSpan(string name)
{
    return Markers.EnterSpan(name);
}

static void WriteProfileFlag(string message)
{
    Markers.WriteFlag(message);
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
    int step = Math.Max(1, (int)Math.Sqrt(n / (double)Superpixels));
    List<Center> centers = CreateCenters(image, step);
    int[] labels = new int[n];

    using Context context = Context.CreateDefault();
    Device device = PickGpuDevice(context);
    using Accelerator accelerator = device.CreateAccelerator(context);
    Console.WriteLine("ILGPU accelerator: " + accelerator);

    using MemoryBuffer1D<float, Stride1D.Dense> dGray = accelerator.Allocate1D(image.Gray);
    using MemoryBuffer1D<float, Stride1D.Dense> dCg = accelerator.Allocate1D<float>(centers.Count);
    using MemoryBuffer1D<float, Stride1D.Dense> dCx = accelerator.Allocate1D<float>(centers.Count);
    using MemoryBuffer1D<float, Stride1D.Dense> dCy = accelerator.Allocate1D<float>(centers.Count);
    using MemoryBuffer1D<int, Stride1D.Dense> dLabels = accelerator.Allocate1D<int>(n);

    var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, SlicViews, SlicSettings>(AssignLabelsKernel);
    float[] cg = new float[centers.Count], cx = new float[centers.Count], cy = new float[centers.Count];

    for (int iter = 0; iter < Iterations; iter++)
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
            new SlicSettings(image.Width, centers.Count, step * 2, Compactness * Compactness / (step * step)));
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
    float tissue = Otsu(image.Gray) * 0.6f;
    int roiLeft = UseBrainRoi ? (int)(image.Width * BrainRoiLeft) : 0;
    int roiTop = UseBrainRoi ? (int)(image.Height * BrainRoiTop) : 0;
    int roiRight = UseBrainRoi ? (int)(image.Width * BrainRoiRight) : image.Width - 1;
    int roiBottom = UseBrainRoi ? (int)(image.Height * BrainRoiBottom) : image.Height - 1;
    double sx = 0, sy = 0;
    int tissuePixels = 0;

    for (int y = 0; y < image.Height; y++)
        for (int x = 0; x < image.Width; x++)
        {
            int i = y * image.Width + x;
            if (image.Gray[i] <= tissue || x < roiLeft || x > roiRight || y < roiTop || y > roiBottom) continue;
            sx += x;
            sy += y;
            tissuePixels++;
        }

    float bx = tissuePixels == 0 ? image.Width / 2f : (float)(sx / tissuePixels);
    float by = tissuePixels == 0 ? image.Height / 2f : (float)(sy / tissuePixels);
    float radius = Math.Min(image.Width, image.Height) * BrainRadiusFactor;
    brain = new BrainInfo(bx, by, radius, tissue, roiLeft, roiTop, roiRight, roiBottom);

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

    means.Sort();
    float mean = count == 0 ? 0 : (float)(sum / count);
    float std = count == 0 ? 0 : MathF.Sqrt(MathF.Max(0, (float)(sumSq / count - mean * mean)));
    float threshold = MathF.Max(mean + TumorSensitivity * std, Percentile(means, TumorPercentile));
    bool[] candidates = new bool[stats.Length];

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
    public void SlicOnlyExecution()
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
        startInfo.ArgumentList.Add("--slic-only");
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
record ImageData(int Width, int Height, byte[] R, byte[] G, byte[] B, float[] Gray);
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
