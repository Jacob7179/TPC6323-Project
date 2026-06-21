using System.Drawing;
using System.Drawing.Imaging;
using System.Diagnostics;
using System.Windows.Forms;
using System.Threading;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;

const int DefaultSuperpixels = 500; // Number of superpixels. Higher = smaller regions, more detail.
const int DefaultIterations = 8; // SLIC update rounds. Higher = more stable, but slower.
const float DefaultCompactness = 10f; // Shape control. Higher = smoother/squarer superpixels, lower = follows edges more.
const float DefaultTumorSensitivity = 0.7f; // Tumor threshold strictness. Higher = fewer highlighted areas, lower = more sensitive.
const float DefaultTumorPercentile = 75f; // Keeps only bright superpixels above this percentile. Higher = fewer candidates.
const float DefaultBrainRadiusFactor = 0.30f; // Removes far outer regions. Lower = stricter against skull/skin edge.
const bool DefaultUseBrainRoi = true; // true = only search inside the brain ROI below.
const float DefaultBrainRoiLeft = 0.56f; // ROI left boundary as image width ratio. Lower if tumor is missed on the left.
const float DefaultBrainRoiTop = 0.41f; // ROI top boundary as image height ratio. Increase to ignore top skull edge.
const float DefaultBrainRoiRight = 0.65f; // ROI right boundary as image width ratio. Lower to ignore right skull edge.
const float DefaultBrainRoiBottom = 0.49f; // ROI bottom boundary as image height ratio. Lower to ignore face/neck areas.
const string InputFolderPath = @"E:\TPC Preprocessing\Preprocess Dataset"; // Put the folder containing preprocessed grayscale images here.
const string OutputFolderPath = @"E:\TPC Project (SLI)\output";
const bool UseGuiPrototype = true; // true = open WinForms GUI, false = run original multiple input folder processing.
const bool RunBenchmarkDotNet = false; // Set true and click Run to execute BenchmarkDotNet instead of normal output generation.
const string EmptyTimingText = "Processing: -- ms\r\nExecution: -- ms";
const string RunningTimingText = "Processing: running...\r\nExecution: running...";

bool runOnce = HasArg(args, "--run-once");
bool runSlicOnly = HasArg(args, "--slic-only");
bool benchmarkRequested = HasArg(args, "--benchmark");
bool runGui = UseGuiPrototype || HasArg(args, "--gui");

if (runGui && !runOnce && !benchmarkRequested)
{
    RunGuiPrototype();
    return;
}

SlicParameters defaultParameters = CreateDefaultParameters();
if (ShouldRunBenchmark(runOnce, benchmarkRequested))
{
    BenchmarkRunner.Run<GpuSlicBenchmark>();
    return;
}

RunBatchProcessing(defaultParameters, runSlicOnly);

static bool HasArg(string[] args, string argument) =>
    args.Contains(argument, StringComparer.OrdinalIgnoreCase);

static bool ShouldRunBenchmark(bool runOnce, bool benchmarkRequested) =>
    benchmarkRequested || (RunBenchmarkDotNet && !runOnce);

static void RunGuiPrototype()
{
    Thread guiThread = new(StartGuiPrototype);
    guiThread.SetApartmentState(ApartmentState.STA);
    guiThread.Start();
    guiThread.Join();
}

static void StartGuiPrototype()
{
    Application.EnableVisualStyles();
    Application.SetCompatibleTextRenderingDefault(false);
    Application.Run(CreatePrototypeForm());
}

static void RunBatchProcessing(SlicParameters defaultParameters, bool runSlicOnly)
{
    string inputFolder = ResolveInputFolder(InputFolderPath);
    string outputDir = PrepareOutputFolder(OutputFolderPath, inputFolder);
    string[] inputPaths = FindInputImages(inputFolder);

    ImageData warmupImage = LoadPreprocessedImage(inputPaths[0]);
    RunGpuSlic(warmupImage, defaultParameters); // Warm-up run for ILGPU kernel compilation. Do not include this in timing.

    BenchmarkSummary benchmark = new();

    foreach (string inputPath in inputPaths)
    {
        ImageData image = LoadAndReportImage(inputPath);
        TimedSlicResult result = RunTimedSlic(image, currentImage => RunGpuSlic(currentImage, defaultParameters));
        benchmark.Add(result.Timing);

        if (runSlicOnly)
            continue;

        SaveTumorCandidateOutput(inputFolder, outputDir, inputPath, image, result.Labels, defaultParameters);
    }

    PrintBenchmark("GPU", benchmark);
}

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

static ImageData LoadAndReportImage(string inputPath)
{
    ImageData image = LoadPreprocessedImage(inputPath);
    Console.WriteLine("Preprocessed image: " + inputPath);
    Console.WriteLine("Size: " + image.Width + " x " + image.Height);
    return image;
}

static TimedSlicResult RunTimedSlic(ImageData image, Func<ImageData, int[]> runSlic)
{
    Stopwatch executionStopwatch = Stopwatch.StartNew();
    Stopwatch processingStopwatch = Stopwatch.StartNew();
    int[] labels = runSlic(image);
    processingStopwatch.Stop();
    executionStopwatch.Stop();

    return new TimedSlicResult(
        labels,
        new SlicTiming(
            processingStopwatch.Elapsed.TotalMilliseconds,
            executionStopwatch.Elapsed.TotalMilliseconds));
}

static void SaveTumorCandidateOutput(string inputFolder, string outputDir, string inputPath, ImageData image, int[] labels, SlicParameters parameters)
{
    bool[] tumorMask = DetectTumorCandidates(image, labels, parameters);
    string relativeFolder = Path.GetDirectoryName(Path.GetRelativePath(inputFolder, inputPath)) ?? "";
    string outputSubFolder = Path.Combine(outputDir, relativeFolder);
    Directory.CreateDirectory(outputSubFolder);

    string outputPath = Path.Combine(
        outputSubFolder,
        Path.GetFileNameWithoutExtension(inputPath) + "_tumor_candidate.jpg");

    SaveTumorOverlay(outputPath, image, tumorMask);
    Console.WriteLine("Output: " + outputPath);
}

static void PrintBenchmark(string label, BenchmarkSummary benchmark)
{
    Console.WriteLine($"{label} SLIC benchmark:");
    Console.WriteLine($"{label} SLIC overall processing time: {benchmark.TotalProcessingTimeMs:F2} ms");
    Console.WriteLine($"{label} SLIC average processing time: {benchmark.AverageProcessingTimeMs:F2} ms");
    Console.WriteLine($"{label} SLIC min processing time: {benchmark.MinProcessingTimeMs:F2} ms");
    Console.WriteLine($"{label} SLIC max processing time: {benchmark.MaxProcessingTimeMs:F2} ms");
    Console.WriteLine($"{label} SLIC overall execution time: {benchmark.TotalExecutionTimeMs:F2} ms");
    Console.WriteLine($"{label} SLIC average execution time: {benchmark.AverageExecutionTimeMs:F2} ms");
    Console.WriteLine($"{label} SLIC min execution time: {benchmark.MinExecutionTimeMs:F2} ms");
    Console.WriteLine($"{label} SLIC max execution time: {benchmark.MaxExecutionTimeMs:F2} ms");
}

static SlicParameters CreateDefaultParameters() => new(
    DefaultSuperpixels,
    DefaultIterations,
    DefaultCompactness,
    DefaultTumorSensitivity,
    DefaultTumorPercentile,
    DefaultBrainRadiusFactor,
    DefaultUseBrainRoi,
    DefaultBrainRoiLeft,
    DefaultBrainRoiTop,
    DefaultBrainRoiRight,
    DefaultBrainRoiBottom);

static Form CreatePrototypeForm()
{
    Form form = new()
    {
        Text = "GPU Parallel SLIC Tumor Highlight Prototype",
        Width = 1280,
        Height = 760,
        MinimumSize = new Size(1050, 650),
        StartPosition = FormStartPosition.CenterScreen,
        KeyPreview = true
    };

    TableLayoutPanel mainLayout = new()
    {
        Dock = DockStyle.Fill,
        ColumnCount = 3,
        RowCount = 1,
        Padding = new Padding(10)
    };
    mainLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 310));
    mainLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));
    mainLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));
    form.Controls.Add(mainLayout);

    FlowLayoutPanel controls = new()
    {
        Dock = DockStyle.Fill,
        FlowDirection = FlowDirection.TopDown,
        WrapContents = false,
        AutoScroll = true,
        Padding = new Padding(8)
    };

    Panel controlPanel = new()
    {
        Dock = DockStyle.Fill,
        Padding = new Padding(4)
    };
    controlPanel.Controls.Add(controls);

    Button uploadButton = new()
    {
        Text = "Upload Image",
        Width = 260,
        Height = 36,
        TabStop = false
    };
    Button startButton = new()
    {
        Text = "Start",
        Width = 260,
        Height = 40,
        Enabled = false
    };
    Label selectedFileLabel = new()
    {
        Text = "No image selected",
        Width = 260,
        Height = 36,
        AutoEllipsis = true
    };
    Label statusLabel = new()
    {
        Text = "Ready",
        Width = 260,
        Height = 50
    };
    Label timingLabel = new()
    {
        Text = EmptyTimingText,
        Dock = DockStyle.Fill,
        TextAlign = ContentAlignment.MiddleRight
    };

    NumericUpDown superpixelsInput = AddParameterControl(controls, "Superpixels", DefaultSuperpixels, 10, 5000, 50, 0);
    NumericUpDown iterationsInput = AddParameterControl(controls, "Iterations", DefaultIterations, 1, 30, 1, 0);
    NumericUpDown compactnessInput = AddParameterControl(controls, "Compactness", (decimal)DefaultCompactness, 0.1m, 100, 0.5m, 1);
    NumericUpDown sensitivityInput = AddParameterControl(controls, "Tumor sensitivity", (decimal)DefaultTumorSensitivity, 0, 5, 0.1m, 1);
    NumericUpDown percentileInput = AddParameterControl(controls, "Tumor percentile", (decimal)DefaultTumorPercentile, 0, 100, 1, 0);
    NumericUpDown radiusInput = AddParameterControl(controls, "Brain radius factor", (decimal)DefaultBrainRadiusFactor, 0.05m, 1, 0.01m, 2);

    CheckBox useBrainRoiInput = new()
    {
        Text = "Use brain ROI",
        Checked = DefaultUseBrainRoi,
        Width = 260,
        Height = 28
    };
    controls.Controls.Add(useBrainRoiInput);

    NumericUpDown roiLeftInput = AddParameterControl(controls, "ROI left", (decimal)DefaultBrainRoiLeft, 0, 1, 0.01m, 2);
    NumericUpDown roiTopInput = AddParameterControl(controls, "ROI top", (decimal)DefaultBrainRoiTop, 0, 1, 0.01m, 2);
    NumericUpDown roiRightInput = AddParameterControl(controls, "ROI right", (decimal)DefaultBrainRoiRight, 0, 1, 0.01m, 2);
    NumericUpDown roiBottomInput = AddParameterControl(controls, "ROI bottom", (decimal)DefaultBrainRoiBottom, 0, 1, 0.01m, 2);
    Label validationLabel = new()
    {
        Text = "",
        Width = 260,
        Height = 64,
        ForeColor = Color.Firebrick,
        Visible = false,
        TextAlign = ContentAlignment.MiddleLeft
    };
    controls.Controls.Add(validationLabel);

    controls.Controls.Add(uploadButton);
    controls.Controls.Add(selectedFileLabel);
    controls.Controls.Add(startButton);
    controls.Controls.Add(statusLabel);

    PictureBox inputPreview = CreatePreviewBox();
    PictureBox outputPreview = CreatePreviewBox();
    mainLayout.Controls.Add(controlPanel, 0, 0);
    mainLayout.Controls.Add(CreateImagePane("Input", inputPreview), 1, 0);
    mainLayout.Controls.Add(CreateImagePane("Output", outputPreview, timingLabel), 2, 0);
    form.AcceptButton = startButton;

    string? selectedPath = null;
    ImageData? selectedImage = null;
    UpdateValidationMessage();

    RegisterValidationEvents(
        UpdateValidationMessage,
        useBrainRoiInput,
        roiLeftInput,
        roiTopInput,
        roiRightInput,
        roiBottomInput);

    uploadButton.Click += (_, _) =>
    {
        using OpenFileDialog dialog = new()
        {
            Filter = "Image files|*.jpg;*.jpeg;*.png;*.bmp|All files|*.*",
            Title = "Select MRI image"
        };

        if (dialog.ShowDialog(form) != DialogResult.OK)
            return;

        try
        {
            selectedPath = dialog.FileName;
            selectedImage = LoadPreprocessedImage(selectedPath);

            ReplacePreviewImage(inputPreview, CreatePreviewBitmap(selectedPath));
            ReplacePreviewImage(outputPreview, null);

            selectedFileLabel.Text = Path.GetFileName(selectedPath);
            statusLabel.Text = "Image loaded";
            timingLabel.Text = EmptyTimingText;
            startButton.Enabled = true;
            startButton.Focus();
        }
        catch (Exception ex)
        {
            selectedPath = null;
            selectedImage = null;
            startButton.Enabled = false;
            MessageBox.Show(form, ex.Message, "Upload failed", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
    };

    startButton.Click += async (_, _) =>
    {
        if (selectedImage is null || string.IsNullOrWhiteSpace(selectedPath))
        {
            MessageBox.Show(form, "Please upload an image first.", "No image", MessageBoxButtons.OK, MessageBoxIcon.Information);
            return;
        }

        try
        {
            UpdateValidationMessage();
            SlicParameters parameters = ReadParameters();
            ImageData imageForRun = selectedImage;
            uploadButton.Enabled = false;
            startButton.Enabled = false;
            form.UseWaitCursor = true;
            statusLabel.Text = "Running GPU SLIC...";
            timingLabel.Text = RunningTimingText;
            Stopwatch executionStopwatch = Stopwatch.StartNew();

            PrototypeRunResult result = await Task.Run(() =>
            {
                Stopwatch processingStopwatch = Stopwatch.StartNew();
                int[] labels = RunGpuSlic(imageForRun, parameters);
                bool[] mask = DetectTumorCandidates(imageForRun, labels, parameters);
                Bitmap output = CreateTumorOverlayBitmap(imageForRun, mask);
                processingStopwatch.Stop();
                return new PrototypeRunResult(output, processingStopwatch.Elapsed.TotalMilliseconds);
            });
            executionStopwatch.Stop();

            ReplacePreviewImage(outputPreview, result.Output);
            timingLabel.Text = $"Processing: {result.ProcessingTimeMs:F2} ms\r\nExecution: {executionStopwatch.Elapsed.TotalMilliseconds:F2} ms";
            statusLabel.Text = "Done";
        }
        catch (Exception ex)
        {
            statusLabel.Text = "Failed";
            MessageBox.Show(form, ex.Message, "Processing failed", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        finally
        {
            form.UseWaitCursor = false;
            uploadButton.Enabled = true;
            startButton.Enabled = selectedImage is not null;
            if (startButton.Enabled)
                startButton.Focus();
        }
    };

    uploadButton.KeyDown += (_, e) => StartOnEnter(e);
    form.KeyDown += (_, e) => StartOnEnter(e);

    form.FormClosed += (_, _) =>
    {
        ReplacePreviewImage(inputPreview, null);
        ReplacePreviewImage(outputPreview, null);
    };

    SlicParameters ReadParameters()
    {
        string validationError = GetValidationMessage();
        if (!string.IsNullOrWhiteSpace(validationError))
            throw new InvalidOperationException(validationError.Replace("\r\n", " "));

        return new SlicParameters(
            (int)superpixelsInput.Value,
            (int)iterationsInput.Value,
            (float)compactnessInput.Value,
            (float)sensitivityInput.Value,
            (float)percentileInput.Value,
            (float)radiusInput.Value,
            useBrainRoiInput.Checked,
            (float)roiLeftInput.Value,
            (float)roiTopInput.Value,
            (float)roiRightInput.Value,
            (float)roiBottomInput.Value);
    }

    void UpdateValidationMessage()
    {
        string validationError = GetValidationMessage();
        validationLabel.Text = validationError;
        validationLabel.Visible = !string.IsNullOrWhiteSpace(validationError);
    }

    string GetValidationMessage()
    {
        if (!useBrainRoiInput.Checked)
            return "";

        List<string> errors = new();
        if (roiLeftInput.Value >= roiRightInput.Value)
            errors.Add("ROI left must be smaller than ROI right.");
        if (roiTopInput.Value >= roiBottomInput.Value)
            errors.Add("ROI top must be smaller than ROI bottom.");

        return errors.Count == 0 ? "" : "Error: " + string.Join("\r\n", errors);
    }

    void StartOnEnter(KeyEventArgs e)
    {
        if (e.KeyCode != Keys.Enter || !startButton.Enabled)
            return;

        e.SuppressKeyPress = true;
        startButton.PerformClick();
    }

    return form;
}

static void RegisterValidationEvents(Action updateValidation, CheckBox checkBox, params NumericUpDown[] inputs)
{
    checkBox.CheckedChanged += (_, _) => updateValidation();
    foreach (NumericUpDown input in inputs)
    {
        input.ValueChanged += (_, _) => updateValidation();
        input.TextChanged += (_, _) => updateValidation();
    }
}

static Bitmap CreatePreviewBitmap(string path)
{
    using Bitmap source = new(path);
    return new Bitmap(source);
}

static void ReplacePreviewImage(PictureBox preview, Image? image)
{
    Image? oldImage = preview.Image;
    preview.Image = image;
    oldImage?.Dispose();
}

static NumericUpDown AddParameterControl(FlowLayoutPanel parent, string labelText, decimal value, decimal min, decimal max, decimal increment, int decimalPlaces)
{
    TableLayoutPanel row = new()
    {
        Width = 260,
        Height = 62,
        ColumnCount = 2,
        RowCount = 2,
        Margin = new Padding(0, 3, 0, 3)
    };
    row.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 60));
    row.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 40));
    row.RowStyles.Add(new RowStyle(SizeType.Absolute, 30));
    row.RowStyles.Add(new RowStyle(SizeType.Absolute, 30));

    Label label = new()
    {
        Text = labelText,
        Dock = DockStyle.Fill,
        TextAlign = ContentAlignment.MiddleLeft
    };
    NumericUpDown input = new NoMouseWheelNumericUpDown()
    {
        Minimum = min,
        Maximum = max,
        Value = value,
        Increment = increment,
        DecimalPlaces = decimalPlaces,
        Dock = DockStyle.Fill
    };
    TrackBar slider = new NoMouseWheelTrackBar()
    {
        Minimum = ToSliderValue(min, decimalPlaces),
        Maximum = ToSliderValue(max, decimalPlaces),
        Value = ToSliderValue(value, decimalPlaces),
        TickStyle = TickStyle.None,
        AutoSize = false,
        Height = 30,
        Dock = DockStyle.Fill
    };
    slider.SmallChange = Math.Max(1, ToSliderValue(increment, decimalPlaces));
    slider.LargeChange = Math.Max(slider.SmallChange, slider.SmallChange * 5);

    bool syncing = false;
    input.ValueChanged += (_, _) =>
    {
        if (syncing) return;
        syncing = true;
        slider.Value = Math.Clamp(ToSliderValue(input.Value, decimalPlaces), slider.Minimum, slider.Maximum);
        syncing = false;
    };
    slider.ValueChanged += (_, _) =>
    {
        if (syncing) return;
        syncing = true;
        input.Value = Math.Clamp(FromSliderValue(slider.Value, decimalPlaces), input.Minimum, input.Maximum);
        syncing = false;
    };

    row.Controls.Add(label, 0, 0);
    row.Controls.Add(input, 1, 0);
    row.Controls.Add(slider, 0, 1);
    row.SetColumnSpan(slider, 2);
    parent.Controls.Add(row);
    return input;
}

static int ToSliderValue(decimal value, int decimalPlaces)
{
    int scale = DecimalScale(decimalPlaces);
    return decimal.ToInt32(decimal.Round(value * scale, MidpointRounding.AwayFromZero));
}

static decimal FromSliderValue(int value, int decimalPlaces)
{
    int scale = DecimalScale(decimalPlaces);
    return value / (decimal)scale;
}

static int DecimalScale(int decimalPlaces)
{
    int scale = 1;
    for (int i = 0; i < decimalPlaces; i++)
        scale *= 10;

    return scale;
}

static PictureBox CreatePreviewBox() => new()
{
    Dock = DockStyle.Fill,
    SizeMode = PictureBoxSizeMode.Zoom,
    BorderStyle = BorderStyle.FixedSingle,
    BackColor = Color.Black
};

static Control CreateImagePane(string title, PictureBox pictureBox, Label? cornerLabel = null)
{
    TableLayoutPanel panel = new()
    {
        Dock = DockStyle.Fill,
        RowCount = 2,
        ColumnCount = 1,
        Padding = new Padding(8)
    };
    panel.RowStyles.Add(new RowStyle(SizeType.Absolute, cornerLabel is null ? 32 : 46));
    panel.RowStyles.Add(new RowStyle(SizeType.Percent, 100));

    TableLayoutPanel header = new()
    {
        Dock = DockStyle.Fill,
        ColumnCount = 2,
        RowCount = 1
    };
    header.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 40));
    header.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 60));
    header.Controls.Add(new Label
    {
        Text = title,
        Dock = DockStyle.Fill,
        TextAlign = ContentAlignment.MiddleLeft,
        Font = new Font(SystemFonts.DefaultFont, FontStyle.Bold)
    }, 0, 0);
    if (cornerLabel is not null)
        header.Controls.Add(cornerLabel, 1, 0);

    panel.Controls.Add(header, 0, 0);
    panel.Controls.Add(pictureBox, 0, 1);
    return panel;
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

    int width = bitmap.Width;
    int height = bitmap.Height;
    int pixelCount = width * height;
    byte[] r = new byte[pixelCount];
    byte[] g = new byte[pixelCount];
    byte[] b = new byte[pixelCount];
    float[] gray = new float[pixelCount];

    BitmapData data = bitmap.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
    try
    {
        unsafe
        {
            byte* ptr = (byte*)data.Scan0;
            for (int y = 0; y < height; y++)
            {
                byte* row = ptr + y * data.Stride;
                for (int x = 0; x < width; x++)
                {
                    int pixelIndex = y * width + x;
                    int byteIndex = x * 3;
                    byte grayValue = row[byteIndex]; // Preprocessing has already made R, G, and B equal.
                    b[pixelIndex] = grayValue;
                    g[pixelIndex] = grayValue;
                    r[pixelIndex] = grayValue;
                    gray[pixelIndex] = grayValue;
                }
            }
        }
    }
    finally
    {
        bitmap.UnlockBits(data);
    }

    return new ImageData(width, height, r, g, b, gray);
}

static Bitmap ConvertTo24BitRgb(Bitmap source)
{
    Bitmap bitmap = new Bitmap(source.Width, source.Height, PixelFormat.Format24bppRgb);
    using Graphics graphics = Graphics.FromImage(bitmap);
    graphics.DrawImage(source, new Rectangle(0, 0, source.Width, source.Height));
    return bitmap;
}

static int[] RunGpuSlic(ImageData image, SlicParameters parameters)
{
    int n = image.Width * image.Height;
    int step = Math.Max(1, (int)Math.Sqrt(n / (double)parameters.Superpixels));
    List<Center> centers = CreateCenters(image, step);
    int[] labels = new int[n];

    using Context context = Context.CreateDefault();
    Device device = PickGpuDevice(context);
    using Accelerator accelerator = device.CreateAccelerator(context);
    Console.WriteLine("ILGPU accelerator: " + accelerator);

    using MemoryBuffer1D<float, Stride1D.Dense> dGray = accelerator.Allocate1D(image.Gray);
    using MemoryBuffer1D<float, Stride1D.Dense> dCenterGray = accelerator.Allocate1D<float>(centers.Count);
    using MemoryBuffer1D<float, Stride1D.Dense> dCenterX = accelerator.Allocate1D<float>(centers.Count);
    using MemoryBuffer1D<float, Stride1D.Dense> dCenterY = accelerator.Allocate1D<float>(centers.Count);
    using MemoryBuffer1D<int, Stride1D.Dense> dLabels = accelerator.Allocate1D<int>(n);

    var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, SlicViews, SlicSettings>(AssignLabelsKernel);
    float[] centerGray = new float[centers.Count];
    float[] centerX = new float[centers.Count];
    float[] centerY = new float[centers.Count];

    for (int iter = 0; iter < parameters.Iterations; iter++)
    {
        CopyCentersToArrays(centers, centerGray, centerX, centerY);

        dCenterGray.CopyFromCPU(centerGray);
        dCenterX.CopyFromCPU(centerX);
        dCenterY.CopyFromCPU(centerY);

        kernel(n, new SlicViews(dGray.View, dCenterGray.View, dCenterX.View, dCenterY.View, dLabels.View),
            new SlicSettings(image.Width, centers.Count, step * 2, parameters.Compactness * parameters.Compactness / (step * step)));
        accelerator.Synchronize();
        dLabels.CopyToCPU(labels);
        UpdateCenters(image, labels, centers);
    }

    return labels;
}

static void CopyCentersToArrays(List<Center> centers, float[] gray, float[] x, float[] y)
{
    for (int i = 0; i < centers.Count; i++)
    {
        gray[i] = centers[i].Gray;
        x[i] = centers[i].X;
        y[i] = centers[i].Y;
    }
}

static Device PickGpuDevice(Context context)
{
    var cuda = context.GetCudaDevices();
    if (cuda.Count > 0)
        return cuda[0];

    var openCl = context.GetCLDevices();
    if (openCl.Count > 0)
        return openCl[0];

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

static bool[] DetectTumorCandidates(ImageData image, int[] labels, SlicParameters parameters)
{
    int labelCount = labels.Max() + 1;
    LabelStats[] stats = BuildLabelStats(image, labels, labelCount, parameters, out BrainInfo brain);
    bool[] candidateLabels = PickBrightCandidateLabels(stats, brain, parameters);
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

static LabelStats[] BuildLabelStats(ImageData image, int[] labels, int labelCount, SlicParameters parameters, out BrainInfo brain)
{
    brain = EstimateBrainInfo(image, parameters);
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
            if (gray > brain.TissueThreshold) s.Tissue++;
            if (x < margin || y < margin || x >= image.Width - margin || y >= image.Height - margin) s.Border++;
            if (TouchesBackground(image, x, y, brain.TissueThreshold)) s.BackgroundNeighbor++;
            if (!InsideBrainRoi(x, y, brain)) s.OutsideRoi++;
            if (Distance(x, y, brain.X, brain.Y) > brain.Radius) s.OutsideRadius++;
        }

    return stats;
}

static BrainInfo EstimateBrainInfo(ImageData image, SlicParameters parameters)
{
    float tissueThreshold = Otsu(image.Gray) * 0.6f;
    int roiLeft = parameters.UseBrainRoi ? (int)(image.Width * parameters.BrainRoiLeft) : 0;
    int roiTop = parameters.UseBrainRoi ? (int)(image.Height * parameters.BrainRoiTop) : 0;
    int roiRight = parameters.UseBrainRoi ? (int)(image.Width * parameters.BrainRoiRight) : image.Width - 1;
    int roiBottom = parameters.UseBrainRoi ? (int)(image.Height * parameters.BrainRoiBottom) : image.Height - 1;
    double sumX = 0;
    double sumY = 0;
    int tissuePixels = 0;

    for (int y = 0; y < image.Height; y++)
        for (int x = 0; x < image.Width; x++)
        {
            int i = y * image.Width + x;
            if (image.Gray[i] <= tissueThreshold || !InsideRoi(x, y, roiLeft, roiTop, roiRight, roiBottom))
                continue;

            sumX += x;
            sumY += y;
            tissuePixels++;
        }

    float brainX = tissuePixels == 0 ? image.Width / 2f : (float)(sumX / tissuePixels);
    float brainY = tissuePixels == 0 ? image.Height / 2f : (float)(sumY / tissuePixels);
    float radius = Math.Min(image.Width, image.Height) * parameters.BrainRadiusFactor;
    return new BrainInfo(brainX, brainY, radius, tissueThreshold, roiLeft, roiTop, roiRight, roiBottom);
}

static bool[] PickBrightCandidateLabels(LabelStats[] stats, BrainInfo brain, SlicParameters parameters)
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
    float threshold = MathF.Max(mean + parameters.TumorSensitivity * std, Percentile(means, parameters.TumorPercentile));
    bool[] candidates = new bool[stats.Length];

    for (int i = 0; i < stats.Length; i++)
        candidates[i] = IsBrainLabel(stats[i], brain) && stats[i].Mean >= threshold;

    return candidates;
}

static bool IsBrainLabel(LabelStats s, BrainInfo brain)
{
    if (s.Count == 0)
        return false;

    return s.TissueRatio >= 0.50f &&
           s.BorderRatio <= 0.30f &&
           s.BackgroundRatio <= 0.35f &&
           s.OutsideRoiRatio <= 0.70f &&
           s.OutsideRatio <= 0.10f &&
           InsideBrainRoi((int)s.CenterX, (int)s.CenterY, brain) &&
           Distance(s.CenterX, s.CenterY, brain.X, brain.Y) <= brain.Radius;
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
            distanceSum += Distance(x, y, brain.X, brain.Y);
            Add(p - 1, x > 0); Add(p + 1, x < w - 1); Add(p - w, y > 0); Add(p + w, y < h - 1);
        }

        int boxW = maxX - minX + 1, boxH = maxY - minY + 1;
        float aspect = Math.Max(boxW, boxH) / (float)Math.Max(1, Math.Min(boxW, boxH));
        float density = region.Count / (float)Math.Max(1, boxW * boxH);
        float meanDistance = distanceSum / region.Count;
        float centerX = sumX / region.Count, centerY = sumY / region.Count;
        bool remove = ShouldRemoveCandidateRegion(
            region.Count,
            minPixels,
            outsideRoiCount,
            centerX,
            centerY,
            meanDistance,
            aspect,
            density,
            brain);

        if (remove)
            ClearRegion(mask, region);

        void Add(int p, bool inside)
        {
            if (!inside || seen[p] || !mask[p]) return;
            seen[p] = true;
            q[tail++] = p;
        }
    }
}

static bool ShouldRemoveCandidateRegion(
    int pixelCount,
    int minPixels,
    int outsideRoiCount,
    float centerX,
    float centerY,
    float meanDistance,
    float aspect,
    float density,
    BrainInfo brain)
{
    return pixelCount < minPixels ||
           !InsideBrainRoi((int)centerX, (int)centerY, brain) ||
           outsideRoiCount > pixelCount * 0.05f ||
           meanDistance > brain.Radius * 0.70f ||
           (aspect > 3f && density < 0.80f) ||
           (meanDistance > brain.Radius * 0.82f && (aspect > 2f || density < 0.65f));
}

static void ClearRegion(bool[] mask, List<int> region)
{
    foreach (int p in region)
        mask[p] = false;
}

static float Distance(float x1, float y1, float x2, float y2)
{
    float dx = x1 - x2;
    float dy = y1 - y2;
    return MathF.Sqrt(dx * dx + dy * dy);
}

static bool InsideBrainRoi(int x, int y, BrainInfo brain)
{
    return InsideRoi(x, y, brain.RoiLeft, brain.RoiTop, brain.RoiRight, brain.RoiBottom);
}

static bool InsideRoi(int x, int y, int left, int top, int right, int bottom)
{
    return x >= left && x <= right && y >= top && y <= bottom;
}

static void SaveTumorOverlay(string path, ImageData image, bool[] mask)
{
    using Bitmap output = CreateTumorOverlayBitmap(image, mask);
    output.Save(path, ImageFormat.Jpeg);
}

static Bitmap CreateTumorOverlayBitmap(ImageData image, bool[] mask)
{
    Bitmap output = new Bitmap(image.Width, image.Height, PixelFormat.Format24bppRgb);
    BitmapData data = output.LockBits(new Rectangle(0, 0, image.Width, image.Height), ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);
    unsafe
    {
        byte* ptr = (byte*)data.Scan0;
        for (int y = 0; y < image.Height; y++)
        {
            byte* row = ptr + y * data.Stride;
            for (int x = 0; x < image.Width; x++)
            {
                int i = y * image.Width + x;
                int p = x * 3;
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
    return output;
}

static bool TouchesBackground(ImageData image, int x, int y, float threshold)
{
    int width = image.Width;
    int height = image.Height;
    int i = y * width + x;

    return x == 0 || y == 0 || x == width - 1 || y == height - 1 ||
           image.Gray[i - 1] <= threshold ||
           image.Gray[i + 1] <= threshold ||
           image.Gray[i - width] <= threshold ||
           image.Gray[i + width] <= threshold;
}

static bool IsMaskBoundary(bool[] mask, int width, int height, int x, int y)
{
    int i = y * width + x;
    return mask[i] &&
           ((x > 0 && !mask[i - 1]) ||
            (x < width - 1 && !mask[i + 1]) ||
            (y > 0 && !mask[i - width]) ||
            (y < height - 1 && !mask[i + width]));
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

record struct SlicTiming(double ProcessingTimeMs, double ExecutionTimeMs);
record struct TimedSlicResult(int[] Labels, SlicTiming Timing);
record struct PrototypeRunResult(Bitmap Output, double ProcessingTimeMs);
record struct SlicParameters(
    int Superpixels,
    int Iterations,
    float Compactness,
    float TumorSensitivity,
    float TumorPercentile,
    float BrainRadiusFactor,
    bool UseBrainRoi,
    float BrainRoiLeft,
    float BrainRoiTop,
    float BrainRoiRight,
    float BrainRoiBottom);

class BenchmarkSummary
{
    public double TotalProcessingTimeMs { get; private set; }
    public double TotalExecutionTimeMs { get; private set; }
    public double MinProcessingTimeMs { get; private set; } = double.MaxValue;
    public double MaxProcessingTimeMs { get; private set; }
    public double MinExecutionTimeMs { get; private set; } = double.MaxValue;
    public double MaxExecutionTimeMs { get; private set; }
    public int Count { get; private set; }

    public double AverageProcessingTimeMs => Count == 0 ? 0 : TotalProcessingTimeMs / Count;
    public double AverageExecutionTimeMs => Count == 0 ? 0 : TotalExecutionTimeMs / Count;

    public void Add(SlicTiming timing)
    {
        Count++;
        TotalProcessingTimeMs += timing.ProcessingTimeMs;
        TotalExecutionTimeMs += timing.ExecutionTimeMs;
        MinProcessingTimeMs = Math.Min(MinProcessingTimeMs, timing.ProcessingTimeMs);
        MaxProcessingTimeMs = Math.Max(MaxProcessingTimeMs, timing.ProcessingTimeMs);
        MinExecutionTimeMs = Math.Min(MinExecutionTimeMs, timing.ExecutionTimeMs);
        MaxExecutionTimeMs = Math.Max(MaxExecutionTimeMs, timing.ExecutionTimeMs);
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

class NoMouseWheelNumericUpDown : NumericUpDown
{
    protected override void OnMouseWheel(MouseEventArgs e)
    {
        if (e is HandledMouseEventArgs handled)
            handled.Handled = true;
    }
}

class NoMouseWheelTrackBar : TrackBar
{
    protected override void OnMouseWheel(MouseEventArgs e)
    {
        if (e is HandledMouseEventArgs handled)
            handled.Handled = true;
    }
}

public readonly record struct SlicViews(
    ArrayView<float> Gray,
    ArrayView<float> CenterGray,
    ArrayView<float> CenterX,
    ArrayView<float> CenterY,
    ArrayView<int> Labels);

public readonly record struct SlicSettings(int Width, int CenterCount, int SearchRadius, float SpatialWeight);
