using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;

// FCM and tumor-candidate settings are chosen automatically per image below.
const int MinFcmClusters = 3;
const int MaxFcmClusters = 5;
const int MinFcmIterations = 8;
const int MaxFcmIterations = 14;
const float Fuzziness = 2.0f;            // Standard FCM fuzziness. Must be greater than 1.

const string InputFolderPath = @"E:\TPC6323-Project\TPC Preprocessing\Preprocess Dataset"; // Put the folder containing preprocessed grayscale images here.
const string OutputFolderPath = @"E:\Fuzzy_C_Means\TPC_Fuzzy_CPU\output";
const string GroundTruthMaskFolderPath = @"E:\TPC6323-Project\Ground Truth Mask";
const bool RunBenchmarkDotNet = false;   // Set true and click Run to execute BenchmarkDotNet instead of normal output generation.

bool runOnce = args.Contains("--run-once", StringComparer.OrdinalIgnoreCase);
if ((RunBenchmarkDotNet && !runOnce) || args.Contains("--benchmark", StringComparer.OrdinalIgnoreCase))
{
    BenchmarkRunner.Run<CpuFcmBenchmark>();
    return;
}

Stopwatch executionStopwatch = Stopwatch.StartNew();
string inputFolder = ResolveInputFolder(InputFolderPath);
string outputDir = PrepareOutputFolder(OutputFolderPath, inputFolder);
string groundTruthFolder = ResolveGroundTruthFolder(GroundTruthMaskFolderPath);
string[] inputPaths = FindInputImages(inputFolder);
BenchmarkSummary benchmark = new();
MseSummary mseSummary = new();

foreach (string inputPath in inputPaths)
{
    ImageRunResult result = ProcessImage(inputFolder, outputDir, groundTruthFolder, inputPath, RunCpuParallelFuzzyCMeans);
    benchmark.Add(result.Timing);
    mseSummary.Add(result.Mse);
    Console.WriteLine("Output: " + result.OutputPath);
    Console.WriteLine(result.Mse.HasValue
        ? $"MSE: {result.Mse.Value:F2}"
        : "MSE: skipped (matching ground truth mask not found)");
}

executionStopwatch.Stop();
PrintBenchmark("CPU", benchmark, executionStopwatch.Elapsed.TotalMilliseconds);
PrintMseSummary("CPU FCM", mseSummary);

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

static string ResolveGroundTruthFolder(string groundTruthFolderPath)
{
    if (string.IsNullOrWhiteSpace(groundTruthFolderPath))
        throw new ArgumentException("Please set GroundTruthMaskFolderPath in Program.cs.");

    string groundTruthFolder = Path.GetFullPath(groundTruthFolderPath);
    if (!Directory.Exists(groundTruthFolder))
        throw new DirectoryNotFoundException("Ground truth mask folder was not found: " + groundTruthFolder);

    return groundTruthFolder;
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

static ImageData LoadAndReportImage(string inputPath)
{
    ImageData image = LoadPreprocessedImage(inputPath);
    Console.WriteLine("Preprocessed image: " + inputPath);
    Console.WriteLine("Size: " + image.Width + " x " + image.Height);
    return image;
}

static ImageRunResult ProcessImage(
    string inputFolder,
    string outputDir,
    string groundTruthFolder,
    string inputPath,
    Func<ImageData, int[]> runSegmentation)
{
    Stopwatch imageExecutionStopwatch = Stopwatch.StartNew();
    ImageData image = LoadAndReportImage(inputPath);

    Stopwatch processingStopwatch = Stopwatch.StartNew();
    int[] labels = runSegmentation(image);
    bool[] tumorMask = DetectTumorCandidates(image, labels);
    processingStopwatch.Stop();

    string outputPath = SaveTumorCandidateOutput(inputFolder, outputDir, inputPath, image, tumorMask);
    imageExecutionStopwatch.Stop();
    double? mse = EvaluateMseAgainstGroundTruth(inputFolder, groundTruthFolder, inputPath, image, tumorMask);

    return new ImageRunResult(
        outputPath,
        new FullBenchmarkTiming(
            processingStopwatch.Elapsed.TotalMilliseconds,
            imageExecutionStopwatch.Elapsed.TotalMilliseconds),
        mse);
}

static string SaveTumorCandidateOutput(
    string inputFolder,
    string outputDir,
    string inputPath,
    ImageData image,
    bool[] tumorMask)
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
    Console.WriteLine($"{label} FCM benchmark:");
    Console.WriteLine($"{label} FCM overall processing time: {benchmark.TotalProcessingTimeMs:F2} ms");
    Console.WriteLine($"{label} FCM average processing time: {benchmark.AverageProcessingTimeMs:F2} ms");
    Console.WriteLine($"{label} FCM min processing time: {benchmark.MinProcessingTimeMs:F2} ms");
    Console.WriteLine($"{label} FCM max processing time: {benchmark.MaxProcessingTimeMs:F2} ms");
    Console.WriteLine($"{label} FCM overall execution time: {executionTimeMs:F2} ms");
    Console.WriteLine($"{label} FCM average execution time: {benchmark.AverageImageExecutionTimeMs:F2} ms");
    Console.WriteLine($"{label} FCM min execution time: {benchmark.MinImageExecutionTimeMs:F2} ms");
    Console.WriteLine($"{label} FCM max execution time: {benchmark.MaxImageExecutionTimeMs:F2} ms");
}

static void PrintMseSummary(string label, MseSummary summary)
{
    Console.WriteLine($"{label} MSE evaluation:");
    Console.WriteLine($"{label} MSE matched images: {summary.Count}");
    Console.WriteLine($"{label} MSE skipped images: {summary.Skipped}");
    if (summary.Count == 0)
        return;

    Console.WriteLine($"{label} MSE average: {summary.Average:F2}");
    Console.WriteLine($"{label} MSE min: {summary.Min:F2}");
    Console.WriteLine($"{label} MSE max: {summary.Max:F2}");
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

    BitmapData data = bitmap.LockBits(
        new Rectangle(0, 0, w, h),
        ImageLockMode.ReadOnly,
        PixelFormat.Format24bppRgb);

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

static double? EvaluateMseAgainstGroundTruth(
    string inputFolder,
    string groundTruthFolder,
    string inputPath,
    ImageData image,
    bool[] predictedMask)
{
    string? groundTruthPath = FindGroundTruthMaskPath(inputFolder, groundTruthFolder, inputPath);
    if (groundTruthPath is null)
        return null;

    return CalculateMse(predictedMask, image.Width, image.Height, groundTruthPath);
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

static Bitmap ConvertTo24BitRgb(Bitmap source)
{
    Bitmap bitmap = new Bitmap(source.Width, source.Height, PixelFormat.Format24bppRgb);
    using Graphics graphics = Graphics.FromImage(bitmap);
    graphics.DrawImage(source, new Rectangle(0, 0, source.Width, source.Height));
    return bitmap;
}

static int[] RunCpuParallelFuzzyCMeans(ImageData image)
{
    if (Fuzziness <= 1f)
        throw new InvalidOperationException("Fuzziness must be greater than 1.");

    int n = image.Width * image.Height;
    AutoFcmSettings settings = ChooseFcmSettings(image);
    int clusterCount = Math.Clamp(settings.Clusters, 1, Math.Max(1, n));
    float[] centers = InitializeFcmCenters(image.Gray, clusterCount);
    int[] labels = new int[n];

    Console.WriteLine("CPU backend: Parallel.For");
    PrintFcmSettings(settings);

    for (int iter = 0; iter < settings.Iterations; iter++)
    {
        double[] globalNumerator = new double[clusterCount];
        double[] globalDenominator = new double[clusterCount];
        object mergeLock = new();

        Parallel.For(
            0,
            n,
            () => new LocalFcmSums(clusterCount),
            (pixel, _, local) =>
            {
                float value = image.Gray[pixel];
                int exactCluster = FindExactCenter(value, centers);

                if (exactCluster >= 0)
                {
                    local.Numerator[exactCluster] += value;
                    local.Denominator[exactCluster] += 1.0;
                    labels[pixel] = exactCluster;
                    return local;
                }

                double bestMembership = -1.0;
                int bestLabel = 0;

                for (int c = 0; c < clusterCount; c++)
                {
                    double membership = CalculateMembership(value, c, centers);
                    double weight = Math.Pow(membership, Fuzziness);
                    local.Numerator[c] += weight * value;
                    local.Denominator[c] += weight;

                    if (membership > bestMembership)
                    {
                        bestMembership = membership;
                        bestLabel = c;
                    }
                }

                labels[pixel] = bestLabel;
                return local;
            },
            local =>
            {
                lock (mergeLock)
                {
                    for (int c = 0; c < clusterCount; c++)
                    {
                        globalNumerator[c] += local.Numerator[c];
                        globalDenominator[c] += local.Denominator[c];
                    }
                }
            });

        for (int c = 0; c < clusterCount; c++)
        {
            if (globalDenominator[c] > 0.0)
                centers[c] = (float)(globalNumerator[c] / globalDenominator[c]);
        }
    }

    AssignHardFcmLabels(image.Gray, centers, labels);
    return labels;
}

static AutoFcmSettings ChooseFcmSettings(ImageData image)
{
    List<float> sorted = new(image.Gray);
    sorted.Sort();
    float spread = Percentile(sorted, 90f) - Percentile(sorted, 10f);
    int clusters = spread >= 115f ? 5 : spread >= 45f ? 4 : 3;
    int iterations = Math.Clamp((int)MathF.Round(8f + spread / 24f), MinFcmIterations, MaxFcmIterations);
    return new AutoFcmSettings(
        Math.Clamp(clusters, MinFcmClusters, MaxFcmClusters),
        iterations,
        Fuzziness);
}

static void PrintFcmSettings(AutoFcmSettings settings)
{
    Console.WriteLine(
        $"Auto FCM settings: clusters={settings.Clusters}, iterations={settings.Iterations}, fuzziness={settings.Fuzziness:F2}");
}

static float[] InitializeFcmCenters(float[] gray, int clusterCount)
{
    int[] hist = new int[256];
    foreach (float value in gray)
        hist[(int)Math.Clamp(value, 0, 255)]++;

    float[] centers = new float[clusterCount];
    int total = gray.Length;
    int cumulative = 0;
    int bucket = 0;

    for (int c = 0; c < clusterCount; c++)
    {
        int target = (int)Math.Round((c + 0.5) * total / clusterCount);
        while (bucket < hist.Length - 1 && cumulative + hist[bucket] < target)
        {
            cumulative += hist[bucket];
            bucket++;
        }

        centers[c] = bucket;
    }

    return centers;
}

static int FindExactCenter(float value, float[] centers)
{
    for (int c = 0; c < centers.Length; c++)
    {
        if (MathF.Abs(value - centers[c]) <= 0.0001f)
            return c;
    }

    return -1;
}

static double CalculateMembership(float value, int cluster, float[] centers)
{
    double distance = FcmDistance(value, centers[cluster]);
    double exponent = 2.0 / (Fuzziness - 1.0);
    double denominator = 0.0;

    for (int c = 0; c < centers.Length; c++)
    {
        double otherDistance = FcmDistance(value, centers[c]);
        denominator += Math.Pow(distance / otherDistance, exponent);
    }

    return 1.0 / denominator;
}

static double FcmDistance(float value, float center)
{
    return Math.Max(0.0001, Math.Abs(value - center));
}

static void AssignHardFcmLabels(float[] gray, float[] centers, int[] labels)
{
    Parallel.For(0, gray.Length, pixel =>
    {
        float value = gray[pixel];
        int bestLabel = 0;
        float bestDistance = float.MaxValue;

        for (int c = 0; c < centers.Length; c++)
        {
            float distance = MathF.Abs(value - centers[c]);
            if (distance < bestDistance)
            {
                bestDistance = distance;
                bestLabel = c;
            }
        }

        labels[pixel] = bestLabel;
    });
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

            bool insideRoi = InsideBrainRoi(x, y, brain);
            if (!insideRoi) s.OutsideRoi++;

            float dx = x - bx, dy = y - by;
            bool outsideRadius = MathF.Sqrt(dx * dx + dy * dy) > radius;
            if (outsideRadius) s.OutsideRadius++;

            if (insideRoi)
            {
                s.RoiSum += gray;
                s.RoiSumSq += gray * gray;
                s.RoiSumX += x;
                s.RoiSumY += y;
                s.RoiCount++;
                if (gray > tissue) s.RoiTissue++;
                if (outsideRadius) s.RoiOutsideRadius++;
            }
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
            if (image.Gray[i] <= tissue)
                continue;

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
            if (image.Gray[i] <= tissue)
                continue;

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
        if (!IsBrainLabel(s, brain))
            continue;

        means.Add(s.RoiMean);
        sum += s.RoiSum;
        sumSq += s.RoiSumSq;
        count += s.RoiCount;
    }

    means.Sort();
    bool[] candidates = new bool[stats.Length];
    if (means.Count == 0)
    {
        PickBrightestFallbackLabel(stats, candidates);
        return candidates;
    }

    float mean = count == 0 ? 0 : (float)(sum / count);
    float std = count == 0 ? 0 : MathF.Sqrt(MathF.Max(0, (float)(sumSq / count - mean * mean)));
    float q25 = Percentile(means, 25f);
    float q75 = Percentile(means, 75f);
    float q90 = Percentile(means, 90f);
    float spread = MathF.Max(1f, q90 - q25);
    float sensitivity = Math.Clamp(spread / 96f, 0.45f, 0.85f);
    float threshold = MathF.Min(q90, MathF.Max(q75, mean + sensitivity * std));

    for (int i = 0; i < stats.Length; i++)
        candidates[i] = IsBrainLabel(stats[i], brain) && stats[i].RoiMean >= threshold;

    if (!candidates.Any(candidate => candidate))
        PickBrightestFallbackLabel(stats, candidates);

    return candidates;
}

static bool IsBrainLabel(LabelStats s, BrainInfo brain)
{
    if (s.RoiCount == 0)
        return false;

    float dx = s.RoiCenterX - brain.X, dy = s.RoiCenterY - brain.Y;
    return s.RoiTissueRatio >= 0.30f &&
           s.RoiOutsideRatio <= 0.65f &&
           InsideBrainRoi((int)s.RoiCenterX, (int)s.RoiCenterY, brain) &&
           MathF.Sqrt(dx * dx + dy * dy) <= brain.Radius;
}

static void PickBrightestFallbackLabel(LabelStats[] stats, bool[] candidates)
{
    int best = -1;
    float bestMean = float.MinValue;

    for (int i = 0; i < stats.Length; i++)
    {
        if (stats[i].RoiCount == 0 || stats[i].RoiTissueRatio < 0.20f)
            continue;

        if (stats[i].RoiMean > bestMean)
        {
            bestMean = stats[i].RoiMean;
            best = i;
        }
    }

    if (best >= 0)
        candidates[best] = true;
}

static void CleanupCandidateRegions(bool[] mask, int w, int h, BrainInfo brain)
{
    bool[] seen = new bool[mask.Length];
    int[] q = new int[mask.Length];
    List<int> region = new();
    int minPixels = Math.Max(32, mask.Length / 3000);

    for (int start = 0; start < mask.Length; start++)
    {
        if (!mask[start] || seen[start])
            continue;

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
            minX = Math.Min(minX, x);
            maxX = Math.Max(maxX, x);
            minY = Math.Min(minY, y);
            maxY = Math.Max(maxY, y);
            sumX += x;
            sumY += y;
            if (!InsideBrainRoi(x, y, brain)) outsideRoiCount++;
            float dx = x - brain.X, dy = y - brain.Y;
            distanceSum += MathF.Sqrt(dx * dx + dy * dy);
            Add(p - 1, x > 0);
            Add(p + 1, x < w - 1);
            Add(p - w, y > 0);
            Add(p + w, y < h - 1);
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

        if (remove)
        {
            foreach (int p in region)
                mask[p] = false;
        }

        void Add(int p, bool inside)
        {
            if (!inside || seen[p] || !mask[p])
                return;

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
    BitmapData data = output.LockBits(
        new Rectangle(0, 0, image.Width, image.Height),
        ImageLockMode.WriteOnly,
        PixelFormat.Format24bppRgb);

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
                if (mask[i])
                {
                    r = Blend(r, 255);
                    g = Blend(g, 255);
                    b = Blend(b, 0);
                }

                if (IsMaskBoundary(mask, image.Width, image.Height, x, y))
                {
                    r = 0;
                    g = 255;
                    b = 0;
                }

                row[p] = b;
                row[p + 1] = g;
                row[p + 2] = r;
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
    return m[i] && ((x > 0 && !m[i - 1]) ||
                    (x < w - 1 && !m[i + 1]) ||
                    (y > 0 && !m[i - w]) ||
                    (y < h - 1 && !m[i + w]));
}

static float Otsu(float[] gray)
{
    int[] hist = new int[256];
    foreach (float v in gray)
        hist[(int)Math.Clamp(v, 0, 255)]++;

    float total = gray.Length, sum = 0, sumB = 0, wB = 0, best = 0;
    int threshold = 0;

    for (int i = 0; i < 256; i++)
        sum += i * hist[i];

    for (int i = 0; i < 256; i++)
    {
        wB += hist[i];
        if (wB == 0)
            continue;

        float wF = total - wB;
        if (wF == 0)
            break;

        sumB += i * hist[i];
        float diff = sumB / wB - (sum - sumB) / wF;
        float between = wB * wF * diff * diff;
        if (between > best)
        {
            best = between;
            threshold = i;
        }
    }

    return threshold;
}

static float Percentile(List<float> sorted, float pct)
{
    if (sorted.Count == 0)
        return float.MaxValue;

    float pos = pct / 100f * (sorted.Count - 1);
    int lo = (int)MathF.Floor(pos), hi = (int)MathF.Ceiling(pos);
    return lo == hi ? sorted[lo] : sorted[lo] * (hi - pos) + sorted[hi] * (pos - lo);
}

static byte Blend(byte original, byte overlay) => (byte)(original * 0.55f + overlay * 0.45f);

public class CpuFcmBenchmark
{
    [Benchmark]
    public void FullPipelineExecution()
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
record struct ImageRunResult(string OutputPath, FullBenchmarkTiming Timing, double? Mse);

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

class LocalFcmSums
{
    public LocalFcmSums(int clusterCount)
    {
        Numerator = new double[clusterCount];
        Denominator = new double[clusterCount];
    }

    public double[] Numerator { get; }
    public double[] Denominator { get; }
}

record ImageData(int Width, int Height, byte[] R, byte[] G, byte[] B, float[] Gray);
record struct AutoFcmSettings(int Clusters, int Iterations, float Fuzziness);
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
    public float Sum, SumSq, SumX, SumY, RoiSum, RoiSumSq, RoiSumX, RoiSumY;
    public int Count, Tissue, Border, BackgroundNeighbor, OutsideRadius, OutsideRoi, RoiCount, RoiTissue, RoiOutsideRadius;
    public float Mean => Count == 0 ? 0 : Sum / Count;
    public float CenterX => Count == 0 ? 0 : SumX / Count;
    public float CenterY => Count == 0 ? 0 : SumY / Count;
    public float TissueRatio => Count == 0 ? 0 : Tissue / (float)Count;
    public float BorderRatio => Count == 0 ? 0 : Border / (float)Count;
    public float BackgroundRatio => Count == 0 ? 0 : BackgroundNeighbor / (float)Count;
    public float OutsideRatio => Count == 0 ? 0 : OutsideRadius / (float)Count;
    public float OutsideRoiRatio => Count == 0 ? 0 : OutsideRoi / (float)Count;
    public float RoiMean => RoiCount == 0 ? 0 : RoiSum / RoiCount;
    public float RoiCenterX => RoiCount == 0 ? 0 : RoiSumX / RoiCount;
    public float RoiCenterY => RoiCount == 0 ? 0 : RoiSumY / RoiCount;
    public float RoiTissueRatio => RoiCount == 0 ? 0 : RoiTissue / (float)RoiCount;
    public float RoiOutsideRatio => RoiCount == 0 ? 0 : RoiOutsideRadius / (float)RoiCount;
}
