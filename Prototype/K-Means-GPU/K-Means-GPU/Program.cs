using System;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.Extensions.DependencyInjection;

class Program
{
    const string WebsiteUrl = "http://localhost:5050";

    static readonly string OutputFolderPath = Path.Combine(AppContext.BaseDirectory, "Output");

    const int K = 3;
    const int BlurRadius = 2;
    const int MaxIterations = 20;
    const int MinBrainRoiPixels = 100;
    const float MinCandidateComponentRatio = 0.00035f;
    const float MaxCandidateComponentRatio = 0.18f;
    const float MaxBrainBoundaryTouchRatio = 0.10f;
    const long MaxUploadBytes = 200L * 1024L * 1024L;

    static async Task Main()
    {
        using GpuKMeansEngine gpu = CreateGpuKMeansEngine();

        await RunWebsiteModeAsync(gpu);
    }

    static GpuKMeansEngine CreateGpuKMeansEngine()
    {
        Context context = Context.Create(builder => builder.Cuda());
        Accelerator accelerator = context.CreateCudaAccelerator(0);
        Console.WriteLine($"Using CUDA GPU: {accelerator.Name}");
        return new GpuKMeansEngine(context, accelerator);
    }

    static async Task RunWebsiteModeAsync(GpuKMeansEngine gpu)
    {
        Directory.CreateDirectory(OutputFolderPath);

        var builder = WebApplication.CreateBuilder();
        builder.WebHost.UseUrls(WebsiteUrl);
        builder.WebHost.ConfigureKestrel(options =>
        {
            options.Limits.MaxRequestBodySize = MaxUploadBytes;
        });
        builder.Services.Configure<FormOptions>(options =>
        {
            options.MultipartBodyLengthLimit = MaxUploadBytes;
        });

        var app = builder.Build();

        app.MapGet("/", () => Results.Content(GetWebsiteHtml(), "text/html"));

        app.MapPost("/segment", async (HttpRequest request) =>
        {
            if (!request.HasFormContentType)
                return Results.BadRequest(new { error = "Expected image upload form." });

            var form = await request.ReadFormAsync();
            if (form.Files.Count == 0)
                return Results.BadRequest(new { error = "No image files uploaded." });

            var results = new List<WebImageResult>();
            foreach (var file in form.Files)
            {
                results.Add(await ProcessUploadedImageAsync(file, gpu));
            }

            return Results.Json(new { results });
        });

        if (!string.Equals(Environment.GetEnvironmentVariable("KMEANS_NO_BROWSER"), "1", StringComparison.OrdinalIgnoreCase))
        {
            _ = OpenBrowserWhenReadyAsync(WebsiteUrl);
        }

        Console.WriteLine($"Web app is running at {WebsiteUrl}");
        await app.RunAsync();
    }

    static async Task<WebImageResult> ProcessUploadedImageAsync(IFormFile file, GpuKMeansEngine gpu)
    {
        try
        {
            if (file.Length == 0)
                return CreateErrorResult(file.FileName, "Empty file.");

            if (!IsSupportedImage(file.FileName))
                return CreateErrorResult(file.FileName, "Only PNG, JPG, and JPEG files are supported.");

            using MemoryStream upload = new();
            await file.CopyToAsync(upload);
            byte[] originalBytes = upload.ToArray();
            upload.Position = 0;

            using Bitmap original = new(upload);
            using Bitmap rgb = ConvertTo24Bit(original);

            using Bitmap overlay = SegmentRgbBitmap(rgb, gpu);
            byte[] outputBytes = BitmapToPngBytes(overlay);

            string safeStem = MakeSafeFileStem(file.FileName);
            string outputPath = GetUniqueOutputPath(safeStem);
            await File.WriteAllBytesAsync(outputPath, outputBytes);

            string tumorType = GetTumorType(file.FileName);

            return new WebImageResult(
                file.FileName,
                ToImageDataUrl(originalBytes, file.FileName),
                ToPngDataUrl(outputBytes),
                outputPath,
                tumorType,
                GetTumorTypeLabel(tumorType),
                null);
        }
        catch (Exception ex)
        {
            return CreateErrorResult(file.FileName, ex.Message);
        }
    }

    static WebImageResult CreateErrorResult(string fileName, string error)
    {
        string tumorType = GetTumorType(fileName);
        return new WebImageResult(
            fileName,
            "",
            "",
            "",
            tumorType,
            GetTumorTypeLabel(tumorType),
            error);
    }

    static async Task OpenBrowserWhenReadyAsync(string url)
    {
        await Task.Delay(800);

        try
        {
            Process.Start(new ProcessStartInfo(url) { UseShellExecute = true });
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Could not open browser automatically: {ex.Message}");
        }
    }

    static Bitmap SegmentRgbBitmap(Bitmap rgb, GpuKMeansEngine gpu)
    {
        int width = rgb.Width;
        int height = rgb.Height;
        float[] gray = GetGrayBytes(rgb);
        gray = GaussianBlurCPU(gray, width, height, BlurRadius);

        bool[] brainMask = BuildBrainRoiMask(gray, width, height);
        int[] roiIndexes = GetMaskIndexes(brainMask);
        if (roiIndexes.Length < MinBrainRoiPixels)
            return CreateGreenOverlayFast(rgb, new bool[width * height]);

        float[] roiPixels = new float[roiIndexes.Length];
        for (int i = 0; i < roiIndexes.Length; i++)
            roiPixels[i] = gray[roiIndexes[i]];

        var (roiLabels, centroids) = KMeansGPU(gpu, roiPixels, roiPixels.Length, 1, K);

        int[] labels = new int[width * height];
        Array.Fill(labels, -1);
        for (int i = 0; i < roiIndexes.Length; i++)
            labels[roiIndexes[i]] = roiLabels[i];

        bool[] candidateMask = BuildTumorCandidateMask(labels, centroids, brainMask, width, height);
        return CreateGreenOverlayFast(rgb, candidateMask);
    }

    static byte[] BitmapToPngBytes(Bitmap bitmap)
    {
        using MemoryStream ms = new();
        bitmap.Save(ms, ImageFormat.Png);
        return ms.ToArray();
    }

    static string ToPngDataUrl(byte[] pngBytes)
    {
        return "data:image/png;base64," + Convert.ToBase64String(pngBytes);
    }

    static string ToImageDataUrl(byte[] imageBytes, string fileName)
    {
        return $"data:{GetImageMimeType(fileName)};base64,{Convert.ToBase64String(imageBytes)}";
    }

    static string GetUniqueOutputPath(string safeStem)
    {
        Directory.CreateDirectory(OutputFolderPath);

        string outputPath = Path.Combine(OutputFolderPath, safeStem + "_tumor.png");
        int index = 1;

        while (File.Exists(outputPath))
        {
            outputPath = Path.Combine(OutputFolderPath, $"{safeStem}_tumor_{index}.png");
            index++;
        }

        return outputPath;
    }

    static string MakeSafeFileStem(string fileName)
    {
        string stem = Path.GetFileNameWithoutExtension(fileName);
        char[] invalid = Path.GetInvalidFileNameChars();
        char[] safeChars = stem
            .Select(ch => invalid.Contains(ch) ? '_' : ch)
            .ToArray();

        string safe = new string(safeChars).Trim();
        return string.IsNullOrWhiteSpace(safe) ? "uploaded_image" : safe;
    }

    static string GetTumorType(string fileName)
    {
        string name = Path.GetFileNameWithoutExtension(fileName).ToLowerInvariant();
        string[] parts = name.Split(
            new[] { '-', '_', ' ', '.', '(', ')', '[', ']' },
            StringSplitOptions.RemoveEmptyEntries);

        if (parts.Contains("gl"))
            return "glioma";

        if (parts.Contains("me"))
            return "meningioma";

        if (parts.Contains("pi"))
            return "pituitary";

        return "unknown";
    }

    static string GetTumorTypeLabel(string tumorType)
    {
        return tumorType switch
        {
            "glioma" => "Glioma",
            "meningioma" => "Meningioma",
            "pituitary" => "Pituitary",
            _ => "Unknown"
        };
    }

    sealed record WebImageResult(
        string fileName,
        string originalDataUrl,
        string outputDataUrl,
        string savedAs,
        string tumorType,
        string tumorTypeLabel,
        string? error);

    static string GetWebsiteHtml() => """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>K-Means Segmentation</title>
    <style>
        :root {
            color-scheme: light;
            --bg: #eef2f4;
            --surface: #ffffff;
            --surface-soft: #f7faf9;
            --ink: #15231f;
            --muted: #65736e;
            --line: #d7e1dd;
            --line-strong: #b9c9c3;
            --accent: #0f766e;
            --accent-dark: #0b534e;
            --accent-soft: #e2f4f1;
            --warm: #b85f2d;
            --warm-soft: #fff0e7;
            --danger: #a73542;
            --shadow: 0 18px 48px rgba(18, 35, 31, .12);
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            min-height: 100vh;
            background-color: var(--bg);
            background-image:
                linear-gradient(rgba(21, 35, 31, .045) 1px, transparent 1px),
                linear-gradient(90deg, rgba(21, 35, 31, .045) 1px, transparent 1px);
            background-size: 34px 34px;
            color: var(--ink);
            font-family: "Segoe UI", Arial, sans-serif;
        }

        body.is-processing .stage {
            box-shadow: 0 20px 54px rgba(15, 118, 110, .16);
        }

        button,
        input {
            font: inherit;
        }

        .hero {
            background:
                linear-gradient(135deg, rgba(13, 40, 37, .98), rgba(15, 118, 110, .92) 56%, rgba(184, 95, 45, .88)),
                repeating-linear-gradient(135deg, rgba(255, 255, 255, .08) 0 1px, transparent 1px 12px);
            color: #ffffff;
            border-bottom: 1px solid rgba(255, 255, 255, .18);
            animation: revealDown .42s ease-out both;
        }

        .hero-inner,
        .shell {
            width: min(1220px, calc(100% - 32px));
            margin: 0 auto;
        }

        .hero-inner {
            display: flex;
            align-items: end;
            justify-content: space-between;
            gap: 24px;
            padding: 30px 0 34px;
            animation: fadeUp .52s ease-out .05s both;
        }

        .eyebrow {
            margin: 0 0 8px;
            color: rgba(255, 255, 255, .76);
            font-size: 12px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0;
        }

        h1 {
            margin: 0;
            max-width: 760px;
            font-size: 38px;
            line-height: 1.08;
            letter-spacing: 0;
        }

        .hero-copy {
            margin: 12px 0 0;
            max-width: 660px;
            color: rgba(255, 255, 255, .82);
            font-size: 15px;
            line-height: 1.55;
        }

        .mode-stack {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            justify-content: flex-end;
        }

        .mode-pill {
            display: inline-flex;
            align-items: center;
            min-height: 34px;
            padding: 0 12px;
            border: 1px solid rgba(255, 255, 255, .26);
            border-radius: 999px;
            background: rgba(255, 255, 255, .12);
            color: #ffffff;
            font-size: 13px;
            font-weight: 700;
            backdrop-filter: blur(10px);
            white-space: nowrap;
            animation: fadeUp .5s ease-out .16s both;
        }

        .shell {
            padding: 20px 0 44px;
        }

        .workspace {
            display: grid;
            grid-template-columns: minmax(300px, 360px) minmax(0, 1fr);
            gap: 18px;
            align-items: start;
        }

        .upload-panel,
        .stage {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: rgba(255, 255, 255, .94);
            box-shadow: var(--shadow);
        }

        .upload-panel {
            position: sticky;
            top: 16px;
            overflow: hidden;
            animation: panelIn .5s ease-out .1s both;
        }

        .panel-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            padding: 16px 16px 12px;
            border-bottom: 1px solid var(--line);
            background: linear-gradient(180deg, #ffffff, #f8fbfa);
        }

        .panel-title {
            margin: 0;
            font-size: 16px;
            line-height: 1.25;
            letter-spacing: 0;
        }

        .file-count {
            display: inline-flex;
            align-items: center;
            min-height: 28px;
            padding: 0 10px;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent-dark);
            font-size: 12px;
            font-weight: 800;
            white-space: nowrap;
        }

        .drop-zone {
            margin: 16px;
            border: 1px dashed var(--line-strong);
            border-radius: 8px;
            background:
                linear-gradient(180deg, rgba(226, 244, 241, .72), rgba(255, 255, 255, .96));
            transition: border-color .18s ease, background .18s ease, transform .18s ease, box-shadow .18s ease;
        }

        .drop-zone:hover {
            border-color: var(--accent);
            box-shadow: inset 0 0 0 1px rgba(15, 118, 110, .08);
        }

        .drop-zone.is-dragging {
            border-color: var(--accent);
            background: var(--accent-soft);
            box-shadow: 0 14px 28px rgba(15, 118, 110, .12);
            transform: translateY(-2px);
        }

        .file-picker {
            position: relative;
            display: grid;
            justify-items: center;
            gap: 10px;
            min-height: 190px;
            padding: 26px 18px;
            text-align: center;
            cursor: pointer;
        }

        .file-picker input {
            position: absolute;
            inset: 0;
            opacity: 0;
            cursor: pointer;
        }

        .upload-mark {
            position: relative;
            width: 54px;
            height: 54px;
            border-radius: 18px;
            background: var(--accent);
            box-shadow: 0 12px 24px rgba(15, 118, 110, .25);
            transition: transform .22s ease, box-shadow .22s ease;
        }

        .drop-zone:hover .upload-mark,
        .drop-zone.is-dragging .upload-mark {
            box-shadow: 0 16px 30px rgba(15, 118, 110, .3);
            transform: translateY(-2px);
        }

        .upload-mark::before,
        .upload-mark::after {
            content: "";
            position: absolute;
            left: 50%;
            background: #ffffff;
            transform: translateX(-50%);
        }

        .upload-mark::before {
            top: 16px;
            width: 4px;
            height: 24px;
            border-radius: 99px;
        }

        .upload-mark::after {
            top: 14px;
            width: 18px;
            height: 18px;
            border-left: 4px solid #ffffff;
            border-top: 4px solid #ffffff;
            background: transparent;
            transform: translateX(-50%) rotate(45deg);
        }

        .picker-main {
            color: var(--ink);
            font-size: 18px;
            font-weight: 800;
            line-height: 1.2;
        }

        .picker-sub {
            color: var(--muted);
            font-size: 13px;
            line-height: 1.45;
        }

        .actions {
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 10px;
            padding: 0 16px 16px;
        }

        button {
            min-height: 44px;
            border: 0;
            border-radius: 8px;
            padding: 0 18px;
            color: #ffffff;
            background: var(--accent);
            font-weight: 800;
            cursor: pointer;
            transition: transform .16s ease, box-shadow .16s ease, opacity .16s ease;
        }

        button:hover:not(:disabled) {
            transform: translateY(-1px);
            box-shadow: 0 12px 22px rgba(15, 118, 110, .22);
        }

        button:disabled {
            opacity: .55;
            cursor: not-allowed;
        }

        button.secondary {
            min-width: 84px;
            border: 1px solid var(--line);
            color: var(--muted);
            background: #ffffff;
        }

        button.secondary:hover:not(:disabled) {
            box-shadow: 0 10px 18px rgba(18, 35, 31, .08);
        }

        .selected-list {
            display: grid;
            gap: 8px;
            padding: 0 16px 16px;
        }

        .selected-item {
            display: grid;
            grid-template-columns: 32px minmax(0, 1fr);
            gap: 9px;
            align-items: center;
            padding: 9px;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--surface-soft);
            animation: listItemIn .24s ease-out both;
        }

        .thumb-mark {
            display: grid;
            place-items: center;
            width: 32px;
            height: 32px;
            border-radius: 8px;
            background: var(--warm-soft);
            color: var(--warm);
            font-size: 12px;
            font-weight: 900;
        }

        .selected-name {
            overflow: hidden;
            color: var(--ink);
            font-size: 13px;
            font-weight: 700;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .selected-size {
            margin-top: 2px;
            color: var(--muted);
            font-size: 12px;
        }

        .status-strip {
            position: relative;
            display: flex;
            align-items: center;
            gap: 10px;
            min-height: 48px;
            margin-bottom: 14px;
            padding: 0 14px;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: rgba(255, 255, 255, .86);
            color: var(--muted);
            overflow: hidden;
            box-shadow: 0 10px 28px rgba(18, 35, 31, .07);
            animation: fadeUp .42s ease-out .16s both;
        }

        .status-strip::after {
            content: "";
            position: absolute;
            inset: auto 0 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--accent), var(--warm), transparent);
            opacity: 0;
            transform: translateX(-100%);
        }

        body.is-processing .status-strip::after {
            animation: statusSweep 1.2s ease-in-out infinite;
            opacity: 1;
        }

        .signal {
            width: 10px;
            height: 10px;
            border-radius: 99px;
            background: var(--line-strong);
        }

        .status-strip[data-state="ready"] .signal {
            background: var(--accent);
        }

        .status-strip[data-state="working"] .signal {
            background: var(--warm);
            animation: pulse 1s ease-in-out infinite;
        }

        .status-strip[data-state="error"] .signal {
            background: var(--danger);
        }

        @keyframes pulse {
            0%, 100% {
                opacity: .45;
                transform: scale(.86);
            }

            50% {
                opacity: 1;
                transform: scale(1.18);
            }
        }

        @keyframes revealDown {
            from {
                opacity: 0;
                transform: translateY(-10px);
            }

            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes fadeUp {
            from {
                opacity: 0;
                transform: translateY(10px);
            }

            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes panelIn {
            from {
                opacity: 0;
                transform: translateY(14px);
            }

            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes listItemIn {
            from {
                opacity: 0;
                transform: translateX(-8px);
            }

            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        @keyframes frameIn {
            from {
                opacity: 0;
                transform: translateY(12px) scale(.992);
            }

            to {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }

        @keyframes imageReveal {
            from {
                opacity: 0;
                transform: scale(.985);
            }

            to {
                opacity: 1;
                transform: scale(1);
            }
        }

        @keyframes softFloat {
            0%, 100% {
                transform: translateY(0);
            }

            50% {
                transform: translateY(-6px);
            }
        }

        @keyframes softFloatLower {
            0%, 100% {
                transform: translateY(12px);
            }

            50% {
                transform: translateY(6px);
            }
        }

        @keyframes statusSweep {
            from {
                transform: translateX(-100%);
            }

            to {
                transform: translateX(100%);
            }
        }

        @keyframes backdropIn {
            from {
                opacity: 0;
            }

            to {
                opacity: 1;
            }
        }

        @keyframes modalPop {
            from {
                opacity: 0;
                transform: translateY(10px) scale(.985);
            }

            to {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }

        .stage {
            min-height: 510px;
            padding: 16px;
            transition: box-shadow .22s ease;
            animation: panelIn .5s ease-out .18s both;
        }

        .stage-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 14px;
            margin-bottom: 14px;
        }

        .stage-title {
            margin: 0;
            font-size: 18px;
            line-height: 1.25;
        }

        .stage-meta {
            color: var(--muted);
            font-size: 13px;
            font-weight: 700;
        }

        .result-tools {
            display: grid;
            grid-template-columns: minmax(220px, 1fr) 180px;
            gap: 10px;
            align-items: end;
            margin-bottom: 14px;
            padding: 12px;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--surface-soft);
        }

        .tool-field {
            display: grid;
            gap: 6px;
            min-width: 0;
        }

        .tool-field span {
            color: var(--muted);
            font-size: 12px;
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: 0;
        }

        .tool-field input,
        .tool-field select {
            width: 100%;
            min-height: 38px;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #ffffff;
            color: var(--ink);
            padding: 0 11px;
            outline: none;
        }

        .tool-field input:focus,
        .tool-field select:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(15, 118, 110, .13);
        }

        .empty-state {
            display: grid;
            place-items: center;
            min-height: 370px;
            border: 1px dashed var(--line-strong);
            border-radius: 8px;
            background:
                linear-gradient(180deg, rgba(247, 250, 249, .95), rgba(255, 255, 255, .98));
            color: var(--muted);
            text-align: center;
            animation: fadeUp .35s ease-out both;
        }

        .empty-state[hidden] {
            display: none;
        }

        .empty-visual {
            display: grid;
            grid-template-columns: repeat(3, 38px);
            gap: 9px;
            justify-content: center;
            margin-bottom: 14px;
        }

        .empty-visual span {
            height: 46px;
            border-radius: 8px;
            background: var(--accent-soft);
            border: 1px solid #c8e6e0;
            animation: softFloat 2.6s ease-in-out infinite;
        }

        .empty-visual span:nth-child(2) {
            background: var(--warm-soft);
            border-color: #f0c9b4;
            transform: translateY(12px);
            animation-name: softFloatLower;
            animation-delay: .18s;
        }

        .empty-visual span:nth-child(3) {
            animation-delay: .34s;
        }

        .empty-state p {
            margin: 0;
            font-size: 14px;
            font-weight: 800;
        }

        .results {
            display: grid;
            gap: 16px;
        }

        .result-frame {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--surface);
            overflow: hidden;
            animation: frameIn .34s ease-out both;
            transition: border-color .18s ease, box-shadow .18s ease, transform .18s ease;
        }

        .result-frame:hover {
            border-color: var(--line-strong);
            box-shadow: 0 18px 38px rgba(18, 35, 31, .1);
            transform: translateY(-1px);
        }

        .result-header {
            display: grid;
            grid-template-columns: auto minmax(0, 1fr) auto;
            gap: 12px;
            align-items: center;
            padding: 14px;
            border-bottom: 1px solid var(--line);
            background: linear-gradient(180deg, #ffffff, #f7faf9);
        }

        .case-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 72px;
            min-height: 34px;
            padding: 0 10px;
            border-radius: 999px;
            background: var(--ink);
            color: #ffffff;
            font-size: 12px;
            font-weight: 900;
            white-space: nowrap;
        }

        .result-title {
            margin: 0;
            font-size: 16px;
            line-height: 1.3;
            overflow-wrap: anywhere;
        }

        .saved-path {
            margin: 3px 0 0;
            color: var(--muted);
            font-size: 12px;
            overflow-wrap: anywhere;
        }

        .result-meta-row {
            display: flex;
            flex-wrap: wrap;
            gap: 7px;
            margin-top: 8px;
        }

        .result-chip {
            display: inline-flex;
            align-items: center;
            min-height: 24px;
            padding: 0 8px;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent-dark);
            font-size: 12px;
            font-weight: 900;
            white-space: nowrap;
        }

        .frame-actions {
            display: flex;
            align-items: center;
            justify-content: flex-end;
            gap: 8px;
            flex-wrap: wrap;
        }

        .download-button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 36px;
            padding: 0 13px;
            border: 1px solid rgba(15, 118, 110, .24);
            border-radius: 8px;
            background: var(--accent-soft);
            color: var(--accent-dark);
            font-size: 13px;
            font-weight: 900;
            text-decoration: none;
            white-space: nowrap;
            transition: transform .16s ease, box-shadow .16s ease, background .16s ease;
        }

        .download-button:hover {
            background: #d5eee9;
            box-shadow: 0 10px 18px rgba(15, 118, 110, .16);
            transform: translateY(-1px);
        }

        .delete-button {
            min-height: 36px;
            border: 1px solid rgba(167, 53, 66, .22);
            border-radius: 8px;
            background: #fff0f1;
            color: var(--danger);
            padding: 0 13px;
            font-size: 13px;
            font-weight: 900;
            box-shadow: none;
        }

        .delete-button:hover:not(:disabled) {
            background: #ffe3e6;
            box-shadow: 0 10px 18px rgba(167, 53, 66, .13);
        }

        .compare {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1px;
            background: var(--line);
        }

        .image-panel {
            min-width: 0;
            background: #f9fbfa;
        }

        .image-label {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            margin: 0;
            padding: 10px 12px;
            border-bottom: 1px solid var(--line);
            color: var(--muted);
            font-size: 12px;
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: 0;
        }

        .image-label::after {
            content: "";
            flex: 0 0 auto;
            width: 8px;
            height: 8px;
            border-radius: 99px;
            background: var(--accent);
        }

        .image-panel.after .image-label::after {
            background: var(--warm);
        }

        .image-box {
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            aspect-ratio: 4 / 3;
            min-height: 250px;
            padding: 14px;
            background:
                linear-gradient(45deg, #eef3f1 25%, transparent 25%),
                linear-gradient(-45deg, #eef3f1 25%, transparent 25%),
                linear-gradient(45deg, transparent 75%, #eef3f1 75%),
                linear-gradient(-45deg, transparent 75%, #eef3f1 75%),
                #f8fbfa;
            background-position: 0 0, 0 10px, 10px -10px, -10px 0;
            background-size: 20px 20px;
            cursor: zoom-in;
            user-select: none;
            -webkit-user-select: none;
            transition: background-color .18s ease;
        }

        .image-box img {
            display: block;
            max-width: 100%;
            max-height: 100%;
            border-radius: 6px;
            object-fit: contain;
            box-shadow: 0 14px 30px rgba(18, 35, 31, .13);
            cursor: zoom-in;
            -webkit-user-drag: none;
            animation: imageReveal .32s ease-out both;
            transition: transform .2s ease, box-shadow .2s ease;
        }

        .image-box:hover img {
            box-shadow: 0 18px 36px rgba(18, 35, 31, .17);
            transform: scale(1.01);
        }

        .frame-zoom-button {
            position: relative;
            width: 36px;
            min-height: 36px;
            height: 36px;
            border: 1px solid rgba(15, 118, 110, .24);
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent-dark);
            padding: 0;
            box-shadow: none;
        }

        .frame-zoom-button::before {
            content: "";
            position: absolute;
            left: 9px;
            top: 8px;
            width: 12px;
            height: 12px;
            border: 2px solid currentColor;
            border-radius: 50%;
        }

        .frame-zoom-button::after {
            content: "";
            position: absolute;
            left: 22px;
            top: 22px;
            width: 10px;
            height: 2px;
            border-radius: 99px;
            background: currentColor;
            transform: rotate(45deg);
            transform-origin: left center;
        }

        .frame-zoom-button:hover:not(:disabled) {
            background: #d5eee9;
            box-shadow: 0 10px 18px rgba(15, 118, 110, .16);
        }

        .lightbox {
            position: fixed;
            inset: 0;
            z-index: 100;
            display: grid;
            place-items: center;
            padding: 34px;
            background: rgba(238, 242, 244, .86);
            backdrop-filter: blur(7px);
            user-select: none;
            -webkit-user-select: none;
            animation: backdropIn .18s ease-out both;
        }

        .lightbox[hidden] {
            display: none;
        }

        .lightbox-content {
            position: relative;
            display: grid;
            place-items: center;
            width: min(1180px, calc(100vw - 74px));
            max-height: calc(100vh - 74px);
        }

        .lightbox-frame {
            display: grid;
            grid-template-rows: auto minmax(0, 1fr);
            width: 100%;
            max-height: calc(100vh - 74px);
            border-radius: 8px;
            background: #ffffff;
            overflow: hidden;
            box-shadow: 0 28px 70px rgba(18, 35, 31, .24);
            animation: modalPop .22s ease-out both;
        }

        .lightbox-frame-head {
            display: grid;
            grid-template-columns: auto minmax(0, 1fr);
            gap: 12px;
            align-items: center;
            padding: 14px;
            border-bottom: 1px solid var(--line);
            background: linear-gradient(180deg, #ffffff, #f7faf9);
        }

        .lightbox-title {
            margin: 0;
            color: var(--ink);
            font-size: 16px;
            line-height: 1.3;
            overflow-wrap: anywhere;
        }

        .lightbox-compare {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1px;
            min-height: 320px;
            overflow: auto;
            background: var(--line);
        }

        .lightbox-panel {
            display: grid;
            grid-template-rows: auto minmax(0, 1fr);
            min-width: 0;
            background: #f9fbfa;
        }

        .lightbox-panel h3 {
            margin: 0;
            padding: 10px 12px;
            border-bottom: 1px solid var(--line);
            color: var(--muted);
            font-size: 12px;
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: 0;
        }

        .lightbox-image-box {
            display: grid;
            place-items: center;
            min-height: 300px;
            padding: 16px;
            background:
                linear-gradient(45deg, #eef3f1 25%, transparent 25%),
                linear-gradient(-45deg, #eef3f1 25%, transparent 25%),
                linear-gradient(45deg, transparent 75%, #eef3f1 75%),
                linear-gradient(-45deg, transparent 75%, #eef3f1 75%),
                #f8fbfa;
            background-position: 0 0, 0 10px, 10px -10px, -10px 0;
            background-size: 20px 20px;
        }

        .lightbox-image-box img {
            display: block;
            max-width: 100%;
            max-height: calc(100vh - 190px);
            border-radius: 6px;
            object-fit: contain;
            box-shadow: 0 16px 34px rgba(18, 35, 31, .16);
            -webkit-user-drag: none;
        }

        .lightbox-close {
            position: absolute;
            top: -18px;
            right: -18px;
            z-index: 2;
            width: 42px;
            min-height: 42px;
            height: 42px;
            border: 2px solid #ffffff;
            border-radius: 999px;
            background: #d92d20;
            color: #ffffff;
            padding: 0;
            box-shadow: 0 14px 28px rgba(167, 53, 66, .28);
            transition: background .16s ease, box-shadow .16s ease, transform .16s ease;
        }

        .lightbox-close:hover,
        .lightbox-close:focus-visible {
            background: #b42318;
            box-shadow: 0 16px 32px rgba(167, 53, 66, .36);
            transform: translateY(-1px);
            outline: none;
        }

        .lightbox-close::before,
        .lightbox-close::after {
            content: "";
            position: absolute;
            left: 12px;
            top: 20px;
            width: 18px;
            height: 2px;
            border-radius: 99px;
            background: currentColor;
        }

        .lightbox-close::before {
            transform: rotate(45deg);
        }

        .lightbox-close::after {
            transform: rotate(-45deg);
        }

        .error {
            margin: 0;
            padding: 16px;
            color: var(--danger);
            font-weight: 800;
        }

        @media (max-width: 920px) {
            .hero-inner {
                align-items: start;
                flex-direction: column;
            }

            .mode-stack {
                justify-content: flex-start;
            }

            .workspace {
                grid-template-columns: 1fr;
            }

            .upload-panel {
                position: static;
            }
        }

        @media (max-width: 720px) {
            .hero-inner,
            .shell {
                width: min(100% - 20px, 1220px);
            }

            .hero-inner {
                padding: 22px 0 26px;
            }

            h1 {
                font-size: 28px;
            }

            .hero-copy {
                font-size: 14px;
            }

            .stage-head,
            .result-header {
                align-items: start;
                grid-template-columns: 1fr;
            }

            .frame-actions {
                justify-self: start;
            }

            .actions,
            .result-tools,
            .compare {
                grid-template-columns: 1fr;
            }

            button.secondary {
                width: 100%;
            }

            .image-box {
                min-height: 210px;
            }

            .lightbox {
                padding: 18px;
            }

            .lightbox-content {
                width: calc(100vw - 36px);
                max-height: calc(100vh - 36px);
            }

            .lightbox-frame {
                max-height: calc(100vh - 36px);
            }

            .lightbox-compare {
                grid-template-columns: 1fr;
            }

            .lightbox-image-box img {
                max-height: calc(100vh - 230px);
            }

            .lightbox-close {
                top: 8px;
                right: 8px;
            }
        }

        @media (prefers-reduced-motion: reduce) {
            *,
            *::before,
            *::after {
                animation-duration: .001ms !important;
                animation-iteration-count: 1 !important;
                scroll-behavior: auto !important;
                transition-duration: .001ms !important;
            }

            .result-frame:hover,
            .image-box:hover img {
                transform: none;
            }
        }
    </style>
</head>
<body>
    <header class="hero">
        <div class="hero-inner">
            <div>
                <p class="eyebrow">ROI-filtered K-Means segmentation</p>
                <h1>Image Segmentation Workspace</h1>
                <p class="hero-copy">Upload MRI images, start the run, and review skull-filtered brain-region candidate highlights.</p>
            </div>
            <div class="mode-stack">
                <span class="mode-pill">GPU parallel</span>
                <span class="mode-pill">Brain ROI filter</span>
            </div>
        </div>
    </header>

    <main class="shell">
        <section class="workspace">
            <aside class="upload-panel">
                <div class="panel-head">
                    <h2 class="panel-title">Upload queue</h2>
                    <span id="fileCount" class="file-count">0 selected</span>
                </div>

                <div id="dropZone" class="drop-zone">
                    <label class="file-picker">
                        <input id="fileInput" type="file" accept=".png,.jpg,.jpeg,image/png,image/jpeg" multiple>
                        <span class="upload-mark" aria-hidden="true"></span>
                        <span class="picker-main">Select images</span>
                        <span class="picker-sub">PNG, JPG, or JPEG</span>
                    </label>
                </div>

                <div class="actions">
                    <button id="startButton" type="button" disabled>Start segmentation</button>
                    <button id="clearButton" class="secondary" type="button">Reset</button>
                </div>

                <div id="selectedList" class="selected-list"></div>
            </aside>

            <section>
                <div id="statusStrip" class="status-strip" data-state="ready" aria-live="polite">
                    <span class="signal" aria-hidden="true"></span>
                    <span id="status">Ready</span>
                </div>

                <section class="stage">
                    <div class="stage-head">
                        <h2 class="stage-title">Segmentation results</h2>
                        <span id="stageMeta" class="stage-meta">No output yet</span>
                    </div>

                    <div class="result-tools">
                        <label class="tool-field">
                            <span>Search</span>
                            <input id="resultSearch" type="search" placeholder="Filename or Frame 01">
                        </label>
                        <label class="tool-field">
                            <span>Tumor type</span>
                            <select id="typeFilter">
                                <option value="all">All types</option>
                                <option value="glioma">Glioma</option>
                                <option value="meningioma">Meningioma</option>
                                <option value="pituitary">Pituitary</option>
                                <option value="unknown">Unknown</option>
                            </select>
                        </label>
                    </div>

                    <div id="emptyState" class="empty-state">
                        <div>
                            <div class="empty-visual" aria-hidden="true">
                                <span></span>
                                <span></span>
                                <span></span>
                            </div>
                            <p id="emptyStateText">Waiting for results</p>
                        </div>
                    </div>

                    <section id="results" class="results" aria-live="polite"></section>
                </section>
            </section>
        </section>
    </main>

    <div id="lightbox" class="lightbox" hidden>
        <div class="lightbox-content">
            <button id="lightboxClose" class="lightbox-close" type="button" aria-label="Close frame preview"></button>
            <section class="lightbox-frame">
                <div class="lightbox-frame-head">
                    <span id="lightboxFrameLabel" class="case-badge">Frame 01</span>
                    <h2 id="lightboxTitle" class="lightbox-title"></h2>
                </div>
                <div class="lightbox-compare">
                    <section class="lightbox-panel">
                        <h3>Before</h3>
                        <div class="lightbox-image-box">
                            <img id="lightboxBefore" alt="Before" draggable="false">
                        </div>
                    </section>
                    <section class="lightbox-panel">
                        <h3>After</h3>
                        <div class="lightbox-image-box">
                            <img id="lightboxAfter" alt="After" draggable="false">
                        </div>
                    </section>
                </div>
            </section>
        </div>
    </div>

    <script>
        const fileInput = document.getElementById('fileInput');
        const dropZone = document.getElementById('dropZone');
        const fileCount = document.getElementById('fileCount');
        const startButton = document.getElementById('startButton');
        const clearButton = document.getElementById('clearButton');
        const statusStrip = document.getElementById('statusStrip');
        const statusBox = document.getElementById('status');
        const stageMeta = document.getElementById('stageMeta');
        const resultSearch = document.getElementById('resultSearch');
        const typeFilter = document.getElementById('typeFilter');
        const selectedList = document.getElementById('selectedList');
        const emptyState = document.getElementById('emptyState');
        const emptyStateText = document.getElementById('emptyStateText');
        const resultsBox = document.getElementById('results');
        const lightbox = document.getElementById('lightbox');
        const lightboxFrameLabel = document.getElementById('lightboxFrameLabel');
        const lightboxTitle = document.getElementById('lightboxTitle');
        const lightboxBefore = document.getElementById('lightboxBefore');
        const lightboxAfter = document.getElementById('lightboxAfter');
        const lightboxClose = document.getElementById('lightboxClose');

        let selectedFiles = [];
        let latestResults = [];
        renderSelectedFiles();

        fileInput.addEventListener('change', () => {
            setSelectedFiles(Array.from(fileInput.files));
        });

        resultSearch.addEventListener('input', renderFilteredResults);
        typeFilter.addEventListener('change', renderFilteredResults);
        lightboxClose.addEventListener('click', closeLightbox);
        lightbox.addEventListener('click', event => {
            if (event.target === lightbox) {
                closeLightbox();
            }
        });
        document.addEventListener('keydown', event => {
            if (event.key === 'Escape' && !lightbox.hidden) {
                closeLightbox();
            }
        });
        document.addEventListener('dragstart', blockImageInteraction);
        document.addEventListener('selectstart', blockImageInteraction);

        for (const eventName of ['dragenter', 'dragover']) {
            dropZone.addEventListener(eventName, event => {
                event.preventDefault();
                dropZone.classList.add('is-dragging');
            });
        }

        for (const eventName of ['dragleave', 'drop']) {
            dropZone.addEventListener(eventName, event => {
                event.preventDefault();
                dropZone.classList.remove('is-dragging');
            });
        }

        dropZone.addEventListener('drop', event => {
            setSelectedFiles(Array.from(event.dataTransfer.files));
        });

        clearButton.addEventListener('click', () => {
            fileInput.value = '';
            setSelectedFiles([]);
            latestResults = [];
            resetResultFilters();
            resultsBox.replaceChildren();
            emptyState.hidden = false;
            emptyStateText.textContent = 'Waiting for results';
            stageMeta.textContent = 'No output yet';
            setStatus('Ready', 'ready');
        });

        startButton.addEventListener('click', async () => {
            const files = selectedFiles;
            if (files.length === 0) {
                setStatus('No images selected.', 'error');
                return;
            }

            const form = new FormData();
            for (const file of files) {
                form.append('images', file);
            }

            startButton.disabled = true;
            clearButton.disabled = true;
            document.body.classList.add('is-processing');
            stageMeta.textContent = 'Running';
            setStatus(`Processing ${files.length} image${files.length === 1 ? '' : 's'}...`, 'working');

            try {
                const response = await fetch('/segment', {
                    method: 'POST',
                    body: form
                });

                const payload = await response.json();
                if (!response.ok) {
                    throw new Error(payload.error || 'Segmentation failed.');
                }

                const results = payload.results || [];
                appendResults(results);
                clearSelectedFilesOnly();
                setStatus(`Added ${results.length} image${results.length === 1 ? '' : 's'}. Total ${latestResults.length} frame${latestResults.length === 1 ? '' : 's'}.`, 'ready');
            } catch (error) {
                renderFilteredResults();
                setStatus(error.message, 'error');
            } finally {
                document.body.classList.remove('is-processing');
                clearButton.disabled = false;
                startButton.disabled = selectedFiles.length === 0;
            }
        });

        function setSelectedFiles(files) {
            selectedFiles = files.filter(isImageFile);

            const transfer = new DataTransfer();
            for (const file of selectedFiles) {
                transfer.items.add(file);
            }
            fileInput.files = transfer.files;

            renderSelectedFiles();
            setStatus(selectedFiles.length === 0 ? 'Ready' : `${selectedFiles.length} image${selectedFiles.length === 1 ? '' : 's'} ready.`, 'ready');
        }

        function renderSelectedFiles() {
            selectedList.replaceChildren();

            const count = selectedFiles.length;
            fileCount.textContent = `${count} selected`;
            startButton.disabled = count === 0;

            for (const [index, file] of selectedFiles.slice(0, 6).entries()) {
                selectedList.appendChild(createSelectedItem(file, index));
            }

            if (selectedFiles.length > 6) {
                const more = document.createElement('div');
                more.className = 'selected-item';
                more.innerHTML = `<span class="thumb-mark">+</span><div><div class="selected-name">${selectedFiles.length - 6} more files</div><div class="selected-size">Queued for segmentation</div></div>`;
                selectedList.appendChild(more);
            }
        }

        function createSelectedItem(file, index) {
            const item = document.createElement('div');
            item.className = 'selected-item';

            const mark = document.createElement('span');
            mark.className = 'thumb-mark';
            mark.textContent = String(index + 1).padStart(2, '0');

            const text = document.createElement('div');

            const name = document.createElement('div');
            name.className = 'selected-name';
            name.textContent = file.name;

            const size = document.createElement('div');
            size.className = 'selected-size';
            size.textContent = formatBytes(file.size);

            text.appendChild(name);
            text.appendChild(size);
            item.appendChild(mark);
            item.appendChild(text);
            return item;
        }

        function appendResults(items) {
            latestResults = latestResults.concat(items);
            renderFilteredResults();
        }

        function clearSelectedFilesOnly() {
            fileInput.value = '';
            selectedFiles = [];
            renderSelectedFiles();
        }

        function renderFilteredResults() {
            resultsBox.replaceChildren();

            const filtered = latestResults
                .map((item, index) => ({ item, index }))
                .filter(entry => matchesSearch(entry.item, entry.index))
                .filter(entry => matchesTumorType(entry.item));

            emptyState.hidden = filtered.length > 0;
            emptyStateText.textContent = latestResults.length === 0 ? 'Waiting for results' : 'No matching results';
            stageMeta.textContent = getStageMetaText(filtered.length, latestResults.length);

            for (const entry of filtered) {
                resultsBox.appendChild(createResultFrame(entry.item, entry.index));
            }
        }

        function createResultFrame(item, index) {
            const frame = document.createElement('article');
            frame.className = 'result-frame';

            const header = document.createElement('div');
            header.className = 'result-header';

            const badge = document.createElement('span');
            badge.className = 'case-badge';
            badge.textContent = getFrameLabel(index);

            const titleBlock = document.createElement('div');
            const title = document.createElement('h3');
            title.className = 'result-title';
            title.textContent = item.fileName;
            titleBlock.appendChild(title);

            if (item.savedAs) {
                const saved = document.createElement('p');
                saved.className = 'saved-path';
                saved.textContent = item.savedAs;
                titleBlock.appendChild(saved);
            }

            if (!item.error) {
                const meta = document.createElement('div');
                meta.className = 'result-meta-row';
                meta.appendChild(createResultChip(item.tumorTypeLabel || 'Unknown'));
                titleBlock.appendChild(meta);
            }

            header.appendChild(badge);
            header.appendChild(titleBlock);

            const actions = document.createElement('div');
            actions.className = 'frame-actions';

            if (!item.error) {
                const zoomFrame = document.createElement('button');
                zoomFrame.className = 'frame-zoom-button';
                zoomFrame.type = 'button';
                zoomFrame.setAttribute('aria-label', `Open ${getFrameLabel(index)} preview`);
                zoomFrame.addEventListener('click', () => openFrameLightbox(item, index));
                actions.appendChild(zoomFrame);
            }

            if (!item.error && item.outputDataUrl) {
                const download = document.createElement('a');
                download.className = 'download-button';
                download.href = item.outputDataUrl;
                download.download = makeDownloadName(item.fileName);
                download.textContent = 'Download result';
                actions.appendChild(download);
            }

            const deleteButton = document.createElement('button');
            deleteButton.className = 'delete-button';
            deleteButton.type = 'button';
            deleteButton.textContent = 'Delete';
            deleteButton.addEventListener('click', () => removeResult(index));
            actions.appendChild(deleteButton);

            header.appendChild(actions);
            frame.appendChild(header);

            if (item.error) {
                const error = document.createElement('p');
                error.className = 'error';
                error.textContent = item.error;
                frame.appendChild(error);
                return frame;
            }

            const compare = document.createElement('div');
            compare.className = 'compare';
            compare.appendChild(createImagePanel('Before', item.originalDataUrl, 'before', () => openFrameLightbox(item, index)));
            compare.appendChild(createImagePanel('After', item.outputDataUrl, 'after', () => openFrameLightbox(item, index)));
            frame.appendChild(compare);

            return frame;
        }

        function removeResult(index) {
            if (index < 0 || index >= latestResults.length) {
                return;
            }

            latestResults.splice(index, 1);
            renderFilteredResults();
            setStatus(latestResults.length === 0 ? 'Ready' : `Removed frame. Total ${latestResults.length} frame${latestResults.length === 1 ? '' : 's'}.`, 'ready');
        }

        function createResultChip(text, extraClass = '') {
            const chip = document.createElement('span');
            chip.className = extraClass ? `result-chip ${extraClass}` : 'result-chip';
            chip.textContent = text;
            return chip;
        }

        function createImagePanel(label, src, type, openFramePreview) {
            const panel = document.createElement('section');
            panel.className = `image-panel ${type}`;

            const heading = document.createElement('h4');
            heading.className = 'image-label';
            heading.textContent = label;
            panel.appendChild(heading);

            const box = document.createElement('div');
            box.className = 'image-box';
            box.addEventListener('click', openFramePreview);

            const image = document.createElement('img');
            image.alt = label;
            image.src = src;
            image.draggable = false;
            box.appendChild(image);

            panel.appendChild(box);
            return panel;
        }

        function openFrameLightbox(item, index) {
            lightboxFrameLabel.textContent = getFrameLabel(index);
            lightboxTitle.textContent = item.fileName;
            lightboxBefore.src = item.originalDataUrl;
            lightboxAfter.src = item.outputDataUrl;
            lightboxBefore.alt = `${item.fileName} before`;
            lightboxAfter.alt = `${item.fileName} after`;
            lightbox.hidden = false;
            lightboxClose.focus();
        }

        function closeLightbox() {
            lightbox.hidden = true;
            lightboxBefore.removeAttribute('src');
            lightboxAfter.removeAttribute('src');
            lightboxBefore.alt = 'Before';
            lightboxAfter.alt = 'After';
            lightboxTitle.textContent = '';
        }

        function blockImageInteraction(event) {
            if (event.target.closest('.image-panel') || event.target.closest('.lightbox')) {
                event.preventDefault();
            }
        }

        function resetResultFilters() {
            resultSearch.value = '';
            typeFilter.value = 'all';
        }

        function matchesSearch(item, index) {
            const query = resultSearch.value.trim().toLowerCase();
            if (!query) {
                return true;
            }

            const searchableText = [
                item.fileName,
                getFrameLabel(index),
                item.tumorTypeLabel
            ].join(' ').toLowerCase();

            return searchableText.includes(query);
        }

        function matchesTumorType(item) {
            return typeFilter.value === 'all' || item.tumorType === typeFilter.value;
        }

        function getStageMetaText(visibleCount, totalCount) {
            if (totalCount === 0) {
                return 'No output yet';
            }

            if (visibleCount === totalCount) {
                return `${totalCount} frame${totalCount === 1 ? '' : 's'}`;
            }

            return `${visibleCount} of ${totalCount} frames`;
        }

        function getFrameLabel(index) {
            return `Frame ${String(index + 1).padStart(2, '0')}`;
        }

        function setStatus(text, state) {
            statusBox.textContent = text;
            statusStrip.dataset.state = state;
        }

        function makeDownloadName(fileName) {
            const clean = fileName.replace(/\.[^.]+$/, '').replace(/[\\/:*?"<>|]+/g, '_');
            return `${clean || 'segmentation_result'}_segmentation.png`;
        }

        function isImageFile(file) {
            return /\.(png|jpe?g)$/i.test(file.name);
        }

        function formatBytes(bytes) {
            if (bytes < 1024) {
                return `${bytes} B`;
            }
            if (bytes < 1024 * 1024) {
                return `${(bytes / 1024).toFixed(1)} KB`;
            }
            return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
        }
    </script>
</body>
</html>
""";

    // =====================================================
    // GPU K-MEANS
    // =====================================================
    sealed class GpuKMeansEngine : IDisposable
    {
        public GpuKMeansEngine(Context context, Accelerator accelerator)
        {
            Context = context;
            Accelerator = accelerator;
            AssignKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>, int>(AssignClustersKernel);
            ReduceKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<int>, int, ArrayView<float>, ArrayView<int>>(CentroidReduceKernel);
            NormalizeKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>>(CentroidNormalizeKernel);
        }

        public Context Context { get; }
        public Accelerator Accelerator { get; }
        public object SyncRoot { get; } = new();
        public Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>, int> AssignKernel { get; }
        public Action<Index1D, ArrayView<float>, ArrayView<int>, int, ArrayView<float>, ArrayView<int>> ReduceKernel { get; }
        public Action<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>> NormalizeKernel { get; }

        public void Dispose()
        {
            Accelerator.Dispose();
            Context.Dispose();
        }
    }

    static (int[] labels, float[] centroids) KMeansGPU(GpuKMeansEngine gpu, float[] pixels, int width, int height, int k)
    {
        int n = width * height;

        float[] centroidsInit = InitializeCentroids(pixels, k);

        lock (gpu.SyncRoot)
        {
            using var gpuPixels = gpu.Accelerator.Allocate1D(pixels);
            using var gpuLabels = gpu.Accelerator.Allocate1D<int>(n);
            using var gpuCentroids = gpu.Accelerator.Allocate1D<float>(centroidsInit);
            using var gpuSums = gpu.Accelerator.Allocate1D<float>(k);
            using var gpuCounts = gpu.Accelerator.Allocate1D<int>(k);

            for (int iter = 0; iter < MaxIterations; iter++)
            {
                gpu.AssignKernel(n, gpuPixels.View, gpuCentroids.View, gpuLabels.View, k);
                gpu.Accelerator.Synchronize();

                gpuSums.MemSetToZero();
                gpuCounts.MemSetToZero();

                gpu.ReduceKernel(n, gpuPixels.View, gpuLabels.View, k, gpuSums.View, gpuCounts.View);
                gpu.Accelerator.Synchronize();

                gpu.NormalizeKernel(k, gpuSums.View, gpuCounts.View, gpuCentroids.View);
                gpu.Accelerator.Synchronize();
            }

            float[] centroids = gpuCentroids.GetAsArray1D();
            int[] labels = gpuLabels.GetAsArray1D();
            return (labels, centroids);
        }
    }

    static float[] InitializeCentroids(float[] pixels, int k)
    {
        float[] centroids = new float[k];
        if (pixels.Length == 0)
            return centroids;

        float min = float.MaxValue;
        float max = float.MinValue;
        foreach (float pixel in pixels)
        {
            if (pixel < min) min = pixel;
            if (pixel > max) max = pixel;
        }

        float range = max - min;
        if (range < 1f || k == 1)
        {
            Array.Fill(centroids, min);
            return centroids;
        }

        for (int i = 0; i < k; i++)
            centroids[i] = min + range * i / (k - 1);

        return centroids;
    }

    static void AssignClustersKernel(Index1D index, ArrayView<float> pixels, ArrayView<float> centroids, ArrayView<int> labels, int k)
    {
        if (index >= pixels.Length)
            return;

        float pix = pixels[index];
        float bestDist = float.MaxValue;
        int bestCluster = 0;

        for (int c = 0; c < k; c++)
        {
            float d = pix - centroids[c];
            d *= d;

            if (d < bestDist)
            {
                bestDist = d;
                bestCluster = c;
            }
        }

        labels[index] = bestCluster;
    }

    static void CentroidReduceKernel(Index1D index, ArrayView<float> pixels, ArrayView<int> labels, int k, ArrayView<float> sums, ArrayView<int> counts)
    {
        if (index >= pixels.Length)
            return;

        int label = labels[index];
        ILGPU.Atomic.Add(ref sums[label], pixels[index]);
        ILGPU.Atomic.Add(ref counts[label], 1);
    }

    static void CentroidNormalizeKernel(Index1D index, ArrayView<float> sums, ArrayView<int> counts, ArrayView<float> centroids)
    {
        if (index >= centroids.Length)
            return;

        int count = counts[index];
        if (count > 0)
            centroids[index] = sums[index] / count;
    }

    // =====================================================
    // PREPROCESSING
    // =====================================================
    static float[] GetGrayBytes(Bitmap bmp)
    {
        int w = bmp.Width, h = bmp.Height;
        float[] gray = new float[w * h];

        BitmapData bmpData = bmp.LockBits(
            new Rectangle(0, 0, w, h),
            ImageLockMode.ReadOnly,
            PixelFormat.Format24bppRgb);

        unsafe
        {
            byte* ptr = (byte*)bmpData.Scan0;

            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    int p = y * bmpData.Stride + x * 3;
                    byte b = ptr[p], g = ptr[p + 1], r = ptr[p + 2];
                    gray[y * w + x] = 0.299f * r + 0.587f * g + 0.114f * b;
                }
        }

        bmp.UnlockBits(bmpData);
        return gray;
    }

    static float[] GaussianBlurCPU(float[] img, int w, int h, int r)
    {
        if (r == 0) return img;

        float sigma = r / 2f;
        float twoSigma2 = 2 * sigma * sigma;

        float[] kernel = new float[r * 2 + 1];
        for (int i = -r; i <= r; i++)
            kernel[i + r] = MathF.Exp(-(i * i) / twoSigma2);

        float sumK = kernel.Sum();
        for (int i = 0; i < kernel.Length; i++)
            kernel[i] /= sumK;

        float[] tmp = new float[w * h];
        float[] res = new float[w * h];

        Parallel.For(0, h, y =>
        {
            for (int x = 0; x < w; x++)
            {
                float s = 0;

                for (int k = -r; k <= r; k++)
                {
                    int xx = Math.Clamp(x + k, 0, w - 1);
                    s += img[y * w + xx] * kernel[k + r];
                }

                tmp[y * w + x] = s;
            }
        });

        Parallel.For(0, h, y =>
        {
            for (int x = 0; x < w; x++)
            {
                float s = 0;

                for (int k = -r; k <= r; k++)
                {
                    int yy = Math.Clamp(y + k, 0, h - 1);
                    s += tmp[yy * w + x] * kernel[k + r];
                }

                res[y * w + x] = s;
            }
        });

        return res;
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
        bool[] gentleErosion = ErodeMask(foreground, w, h, fallbackRadius);
        gentleErosion = KeepLargestComponent(gentleErosion, w, h);
        return CountMask(gentleErosion) >= MinBrainRoiPixels
            ? gentleErosion
            : new bool[w * h];
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
        int tumorCluster = FindTumorCluster(labels, centroids, brainMask, w, h);
        bool[] rawCandidate = new bool[w * h];
        if (tumorCluster < 0)
            return rawCandidate;

        for (int i = 0; i < rawCandidate.Length; i++)
            rawCandidate[i] = brainMask[i] && labels[i] == tumorCluster;

        return CleanCandidateComponents(rawCandidate, brainMask, w, h);
    }

    static int FindTumorCluster(int[] labels, float[] centroids, bool[] brainMask, int w, int h)
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

    static int[] GetMaskIndexes(bool[] mask)
    {
        int count = CountMask(mask);
        int[] indexes = new int[count];
        int write = 0;

        for (int i = 0; i < mask.Length; i++)
        {
            if (mask[i])
                indexes[write++] = i;
        }

        return indexes;
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

        Parallel.For(0, h, y =>
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
        });

        return result;
    }

    static bool[] ErodeMask(bool[] mask, int w, int h, int radius)
    {
        if (radius <= 0)
            return (bool[])mask.Clone();

        bool[] result = new bool[mask.Length];

        Parallel.For(0, h, y =>
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
        });

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

    // =====================================================
    // UTILITIES (unchanged)
    // =====================================================
    static bool IsSupportedImage(string path)
    {
        string ext = Path.GetExtension(path).ToLowerInvariant();
        return ext is ".jpg" or ".jpeg" or ".png";
    }

    static string GetImageMimeType(string path)
    {
        string ext = Path.GetExtension(path).ToLowerInvariant();
        return ext == ".png" ? "image/png" : "image/jpeg";
    }

    static Bitmap ConvertTo24Bit(Bitmap src)
    {
        Bitmap bmp = new(src.Width, src.Height, PixelFormat.Format24bppRgb);
        CopyResolution(src, bmp);
        using Graphics g = Graphics.FromImage(bmp);
        g.DrawImage(src, 0, 0);
        return bmp;
    }

    static void CopyResolution(Image source, Bitmap target)
    {
        if (source.HorizontalResolution > 0 && source.VerticalResolution > 0)
        {
            target.SetResolution(source.HorizontalResolution, source.VerticalResolution);
        }
    }

    static unsafe Bitmap CreateGreenOverlayFast(Bitmap original, bool[] highlightMask)
    {
        int w = original.Width, h = original.Height;
        Bitmap result = new(w, h, PixelFormat.Format24bppRgb);
        CopyResolution(original, result);

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

        Parallel.For(0, h, y =>
        {
            byte* srow = sp + y * src.Stride;
            byte* drow = dp + y * dst.Stride;
            int baseIdx = y * w;

            for (int x = 0; x < w; x++)
            {
                int i = baseIdx + x;
                int p = x * 3;

                byte b = srow[p], g = srow[p + 1], r = srow[p + 2];
                byte gray = (byte)(0.299f * r + 0.587f * g + 0.114f * b);

                if (highlightMask[i])
                {
                    drow[p] = (byte)(gray * 0.3f);
                    drow[p + 1] = 255;
                    drow[p + 2] = (byte)(gray * 0.3f);
                }
                else
                {
                    drow[p] = gray;
                    drow[p + 1] = gray;
                    drow[p + 2] = gray;
                }
            }
        });

        original.UnlockBits(src);
        result.UnlockBits(dst);
        return result;
    }
}
