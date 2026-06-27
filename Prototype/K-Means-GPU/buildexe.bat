@echo off
dotnet publish "E:\Prototype\K-Means-GPU\K-Means-GPU\K-Means-GPU.csproj" -c Release -r win-x64 --self-contained true -o "E:\Prototype\K-Means-GPU\PublishedExe" /p:PublishSingleFile=true /p:IncludeNativeLibrariesForSelfExtract=true /p:DebugSymbols=false /p:AssemblyName=BrainTumorSegmentation
pause