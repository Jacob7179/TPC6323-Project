using System.Drawing;
using System.Drawing.Imaging;

const string InputFolderPath = @"E:\TPC6323-Project\TPC Preprocessing\Dataset"; // Put the folder containing original MRI images here.
const string OutputFolderPath = @"E:\TPC6323-Project\TPC Preprocessing\Preprocess Dataset";

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

foreach (string inputPath in inputPaths)
{
    ImageData image = LoadImage(inputPath);
    Console.WriteLine("Image: " + inputPath);
    Console.WriteLine("Size: " + image.Width + " x " + image.Height);

    string relativeFolder = Path.GetDirectoryName(Path.GetRelativePath(inputFolder, inputPath)) ?? "";
    string outputSubFolder = Path.Combine(outputDir, relativeFolder);
    Directory.CreateDirectory(outputSubFolder);

    string outputPath = Path.Combine(
        outputSubFolder,
        Path.GetFileNameWithoutExtension(inputPath) + "_gray.png");

    SaveGrayscaleImage(outputPath, image);
    Console.WriteLine("Output: " + outputPath);
}

static bool IsSupportedImage(string path)
{
    string ext = Path.GetExtension(path).ToLowerInvariant();
    return ext is ".jpg" or ".jpeg" or ".png" or ".bmp";
}

static ImageData LoadImage(string path)
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
                b[i] = row[p];
                g[i] = row[p + 1];
                r[i] = row[p + 2];
                gray[i] = (r[i] + g[i] + b[i]) / 3f;
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

static void SaveGrayscaleImage(string path, ImageData image)
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
                byte gray = (byte)Math.Clamp((int)image.Gray[i], 0, 255);
                row[p] = gray;
                row[p + 1] = gray;
                row[p + 2] = gray;
            }
        }
    }

    output.UnlockBits(data);
    output.Save(path, ImageFormat.Png);
}

record ImageData(int Width, int Height, byte[] R, byte[] G, byte[] B, float[] Gray);
