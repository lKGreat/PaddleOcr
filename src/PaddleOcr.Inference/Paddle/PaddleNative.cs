using System.Runtime.InteropServices;

namespace PaddleOcr.Inference.Paddle;

public sealed class PaddleNative : IDisposable
{
    private static readonly object DllSearchLock = new();
    private static bool _dllSearchPrepared;
    private static readonly List<IntPtr> DependencyHandles = new();

    private readonly IntPtr _library;
    private readonly Api _api;
    private bool _disposed;

    private PaddleNative(IntPtr library, Api api)
    {
        _library = library;
        _api = api;
    }

    public static PaddleNative Create(string? paddleLibDir)
    {
        var library = LoadLibrary(paddleLibDir);
        var api = Api.Load(library);
        return new PaddleNative(library, api);
    }

    public PaddlePredictor CreatePredictor(string modelDirOrFile, string? modelParamsFile = null)
    {
        EnsureNotDisposed();
        var (graphPath, paramsPath) = ResolveModelArtifacts(modelDirOrFile, modelParamsFile);
        return new PaddlePredictor(_api, graphPath, paramsPath);
    }

    private static IntPtr LoadLibrary(string? paddleLibDir)
    {
        PrepareNativeSearchPaths(paddleLibDir);

        var candidateNames = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
            ? new[] { "paddle_inference_c.dll", "paddle_inference.dll", "paddle.dll" }
            : RuntimeInformation.IsOSPlatform(OSPlatform.OSX)
                ? new[] { "libpaddle_inference_c.dylib", "libpaddle_inference_c.so", "libpaddle_inference.dylib", "libpaddle_inference.so" }
                : new[] { "libpaddle_inference_c.so", "libpaddle_inference.so" };

        var candidates = new List<string>(candidateNames.Length * 2);
        if (!string.IsNullOrWhiteSpace(paddleLibDir))
        {
            candidates.AddRange(candidateNames.Select(x => Path.Combine(paddleLibDir, x)));
        }

        candidates.AddRange(candidateNames);

        foreach (var candidate in candidates.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            try
            {
                if (NativeLibrary.TryLoad(candidate, out var handle))
                {
                    return handle;
                }
            }
            catch
            {
                // Try next candidate.
            }
        }

        throw new DllNotFoundException(
            "Unable to load Paddle Inference C library. " +
            "Expected paddle_inference_c runtime (e.g. paddle_inference_c.dll / libpaddle_inference_c.so).");
    }

    private static void PrepareNativeSearchPaths(string? paddleLibDir)
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            return;
        }

        lock (DllSearchLock)
        {
            if (_dllSearchPrepared)
            {
                return;
            }

            var dirs = new List<string>();
            if (!string.IsNullOrWhiteSpace(paddleLibDir) && Directory.Exists(paddleLibDir))
            {
                dirs.Add(paddleLibDir);

                var libDir = new DirectoryInfo(paddleLibDir);
                var paddleDir = libDir.Parent;
                var rootDir = paddleDir?.Parent;
                if (rootDir is not null)
                {
                    var mklml = Path.Combine(rootDir.FullName, "third_party", "install", "mklml", "lib");
                    var onednn = Path.Combine(rootDir.FullName, "third_party", "install", "onednn", "lib");
                    if (Directory.Exists(mklml))
                    {
                        dirs.Add(mklml);
                    }

                    if (Directory.Exists(onednn))
                    {
                        dirs.Add(onednn);
                    }
                }
            }

            var vcomp = ResolveVcomp140Path();
            if (!string.IsNullOrWhiteSpace(vcomp))
            {
                dirs.Add(Path.GetDirectoryName(vcomp)!);
            }

            TrySetDefaultDllDirectories();
            foreach (var dir in dirs.Distinct(StringComparer.OrdinalIgnoreCase))
            {
                TryAddDllDirectory(dir);
            }

            TryPreloadDependencies(dirs, vcomp);
            _dllSearchPrepared = true;
        }
    }

    private static void TryPreloadDependencies(IReadOnlyList<string> dirs, string? vcompPath)
    {
        var preloadNames = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
            ? new[] { "common.dll", "libiomp5md.dll", "mklml.dll", "mkldnn.dll" }
            : Array.Empty<string>();
        foreach (var dir in dirs)
        {
            foreach (var name in preloadNames)
            {
                var full = Path.Combine(dir, name);
                TryLoadDependency(full);
            }
        }

        if (!string.IsNullOrWhiteSpace(vcompPath))
        {
            TryLoadDependency(vcompPath);
        }
        else
        {
            TryLoadDependency("vcomp140.dll");
        }
    }

    private static void TryLoadDependency(string fileOrName)
    {
        try
        {
            if (NativeLibrary.TryLoad(fileOrName, out var handle) && handle != IntPtr.Zero)
            {
                DependencyHandles.Add(handle);
            }
        }
        catch
        {
            // Ignore. Main load path will surface a clear error if unresolved.
        }
    }

    private static string? ResolveVcomp140Path()
    {
        try
        {
            var programFiles = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
            if (string.IsNullOrWhiteSpace(programFiles))
            {
                return null;
            }

            var vsRoot = Path.Combine(programFiles, "Microsoft Visual Studio");
            if (!Directory.Exists(vsRoot))
            {
                return null;
            }

            return Directory
                .EnumerateFiles(vsRoot, "vcomp140.dll", SearchOption.AllDirectories)
                .FirstOrDefault();
        }
        catch
        {
            return null;
        }
    }

    private static void TrySetDefaultDllDirectories()
    {
        try
        {
            _ = SetDefaultDllDirectories(LoadLibrarySearchDefaultDirs | LoadLibrarySearchUserDirs);
        }
        catch
        {
            // Ignore on unsupported systems.
        }
    }

    private static void TryAddDllDirectory(string dir)
    {
        if (string.IsNullOrWhiteSpace(dir) || !Directory.Exists(dir))
        {
            return;
        }

        try
        {
            _ = AddDllDirectory(dir);
        }
        catch
        {
            // Ignore and continue.
        }
    }

    private static (string GraphPath, string ParamsPath) ResolveModelArtifacts(string modelDirOrFile, string? modelParamsFile)
    {
        if (Directory.Exists(modelDirOrFile))
        {
            var graph = FirstExisting(
                Path.Combine(modelDirOrFile, "inference.json"),
                Path.Combine(modelDirOrFile, "inference.pdmodel"),
                Path.Combine(modelDirOrFile, "model.json"),
                Path.Combine(modelDirOrFile, "model.pdmodel"));
            var parms = FirstExisting(
                Path.Combine(modelDirOrFile, "inference.pdiparams"),
                Path.Combine(modelDirOrFile, "model.pdiparams"),
                Path.Combine(modelDirOrFile, "model.pdparams"));
            if (graph is not null && parms is not null)
            {
                return (graph, parms);
            }

            throw new FileNotFoundException($"Paddle model dir missing inference graph/params: {modelDirOrFile}");
        }

        if (!File.Exists(modelDirOrFile))
        {
            throw new FileNotFoundException($"Paddle model source not found: {modelDirOrFile}");
        }

        var ext = Path.GetExtension(modelDirOrFile).ToLowerInvariant();
        if (ext is not (".json" or ".pdmodel"))
        {
            throw new InvalidOperationException(
                $"Unsupported Paddle model file extension: {modelDirOrFile}. expected .json or .pdmodel");
        }

        if (!string.IsNullOrWhiteSpace(modelParamsFile) && File.Exists(modelParamsFile))
        {
            return (modelDirOrFile, modelParamsFile);
        }

        var dir = Path.GetDirectoryName(modelDirOrFile) ?? Directory.GetCurrentDirectory();
        var baseName = Path.Combine(dir, Path.GetFileNameWithoutExtension(modelDirOrFile));
        var inferred = FirstExisting(
            baseName + ".pdiparams",
            Path.Combine(dir, "inference.pdiparams"),
            Path.Combine(dir, "model.pdiparams"),
            Path.Combine(dir, "model.pdparams"));
        if (inferred is null)
        {
            throw new FileNotFoundException($"Cannot infer pdiparams for model file: {modelDirOrFile}");
        }

        return (modelDirOrFile, inferred);
    }

    private static string? FirstExisting(params string[] files)
    {
        return files.FirstOrDefault(File.Exists);
    }

    private void EnsureNotDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(PaddleNative));
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        NativeLibrary.Free(_library);
        _disposed = true;
    }

    public sealed class PaddlePredictor : IDisposable
    {
        private readonly Api _api;
        private readonly IntPtr _predictor;
        private readonly string[] _inputNames;
        private readonly string[] _outputNames;
        private bool _disposed;

        internal PaddlePredictor(Api api, string graphPath, string paramsPath)
        {
            _api = api;
            var config = _api.ConfigCreate();
            if (config == IntPtr.Zero)
            {
                throw new InvalidOperationException("PD_ConfigCreate returned null.");
            }

            try
            {
                _api.ConfigSetModel(config, graphPath, paramsPath);
                _api.ConfigDisableGpu(config);
                _predictor = _api.PredictorCreate(config);
            }
            finally
            {
                // PredictorCreate may take ownership, but explicit destroy is safe for null/non-owned configs in practice.
                _api.ConfigDestroy(config);
            }

            if (_predictor == IntPtr.Zero)
            {
                throw new InvalidOperationException("PD_PredictorCreate returned null.");
            }

            _inputNames = ReadCstrArray(_api.PredictorGetInputNames(_predictor), ptr => _api.OneDimArrayCstrDestroy(ptr));
            _outputNames = ReadCstrArray(_api.PredictorGetOutputNames(_predictor), ptr => _api.OneDimArrayCstrDestroy(ptr));
            if (_inputNames.Length == 0 || _outputNames.Length == 0)
            {
                throw new InvalidOperationException("Paddle predictor has empty input/output names.");
            }
        }

        public bool HasInput(string name)
        {
            return _inputNames.Any(x => x.Equals(name, StringComparison.Ordinal));
        }

        public (float[] Data, int[] Dims) Run(float[] inputData, int[] inputDims, float? validRatio = null)
        {
            EnsureNotDisposed();
            if (inputDims.Length == 0)
            {
                throw new ArgumentException("input dims cannot be empty", nameof(inputDims));
            }

            var inputHandle = _api.PredictorGetInputHandle(_predictor, _inputNames[0]);
            if (inputHandle == IntPtr.Zero)
            {
                throw new InvalidOperationException($"Failed to get input handle: {_inputNames[0]}");
            }

            try
            {
                _api.TensorReshape(inputHandle, (nuint)inputDims.Length, inputDims);
                _api.TensorCopyFromCpuFloat(inputHandle, inputData);
            }
            finally
            {
                _api.TensorDestroy(inputHandle);
            }

            if (validRatio.HasValue && HasInput("valid_ratio"))
            {
                var ratioHandle = _api.PredictorGetInputHandle(_predictor, "valid_ratio");
                if (ratioHandle != IntPtr.Zero)
                {
                    try
                    {
                        _api.TensorReshape(ratioHandle, 1, [1]);
                        _api.TensorCopyFromCpuFloat(ratioHandle, [validRatio.Value]);
                    }
                    finally
                    {
                        _api.TensorDestroy(ratioHandle);
                    }
                }
            }

            var ok = _api.PredictorRun(_predictor);
            if (ok == 0)
            {
                throw new InvalidOperationException("PD_PredictorRun returned false.");
            }

            var outputHandle = _api.PredictorGetOutputHandle(_predictor, _outputNames[0]);
            if (outputHandle == IntPtr.Zero)
            {
                throw new InvalidOperationException($"Failed to get output handle: {_outputNames[0]}");
            }

            try
            {
                var dims = ReadInt32Array(_api.TensorGetShape(outputHandle), ptr => _api.OneDimArrayInt32Destroy(ptr));
                if (dims.Length == 0)
                {
                    throw new InvalidOperationException("Output shape is empty.");
                }

                var total = 1;
                foreach (var d in dims)
                {
                    total = checked(total * Math.Max(1, d));
                }

                var output = new float[total];
                _api.TensorCopyToCpuFloat(outputHandle, output);
                return (output, dims);
            }
            finally
            {
                _api.TensorDestroy(outputHandle);
            }
        }

        private static string[] ReadCstrArray(IntPtr arrayPtr, Action<IntPtr> destroy)
        {
            if (arrayPtr == IntPtr.Zero)
            {
                return [];
            }

            try
            {
                var array = Marshal.PtrToStructure<OneDimArrayCstr>(arrayPtr);
                var count = checked((int)array.Size);
                var names = new string[count];
                for (var i = 0; i < count; i++)
                {
                    var ptr = Marshal.ReadIntPtr(array.Data, i * IntPtr.Size);
                    names[i] = Marshal.PtrToStringUTF8(ptr) ?? string.Empty;
                }

                return names.Where(x => !string.IsNullOrWhiteSpace(x)).ToArray();
            }
            finally
            {
                destroy(arrayPtr);
            }
        }

        private static int[] ReadInt32Array(IntPtr arrayPtr, Action<IntPtr> destroy)
        {
            if (arrayPtr == IntPtr.Zero)
            {
                return [];
            }

            try
            {
                var array = Marshal.PtrToStructure<OneDimArrayInt32>(arrayPtr);
                var count = checked((int)array.Size);
                var dims = new int[count];
                for (var i = 0; i < count; i++)
                {
                    dims[i] = Marshal.ReadInt32(array.Data, i * sizeof(int));
                }

                return dims;
            }
            finally
            {
                destroy(arrayPtr);
            }
        }

        private void EnsureNotDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(PaddlePredictor));
            }
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            if (_predictor != IntPtr.Zero)
            {
                _api.PredictorDestroy(_predictor);
            }

            _disposed = true;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    private readonly struct OneDimArrayCstr
    {
        public readonly nuint Size;
        public readonly IntPtr Data;
    }

    [StructLayout(LayoutKind.Sequential)]
    private readonly struct OneDimArrayInt32
    {
        public readonly nuint Size;
        public readonly IntPtr Data;
    }

    private const uint LoadLibrarySearchDefaultDirs = 0x00001000;
    private const uint LoadLibrarySearchUserDirs = 0x00000400;

    [DllImport("kernel32", SetLastError = true)]
    private static extern bool SetDefaultDllDirectories(uint directoryFlags);

    [DllImport("kernel32", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr AddDllDirectory(string newDirectory);

    internal sealed class Api
    {
        public required ConfigCreateDelegate ConfigCreate { get; init; }
        public required ConfigDestroyDelegate ConfigDestroy { get; init; }
        public required ConfigSetModelDelegate ConfigSetModel { get; init; }
        public required ConfigDisableGpuDelegate ConfigDisableGpu { get; init; }
        public required PredictorCreateDelegate PredictorCreate { get; init; }
        public required PredictorDestroyDelegate PredictorDestroy { get; init; }
        public required PredictorGetInputNamesDelegate PredictorGetInputNames { get; init; }
        public required PredictorGetOutputNamesDelegate PredictorGetOutputNames { get; init; }
        public required PredictorGetInputHandleDelegate PredictorGetInputHandle { get; init; }
        public required PredictorGetOutputHandleDelegate PredictorGetOutputHandle { get; init; }
        public required PredictorRunDelegate PredictorRun { get; init; }
        public required TensorReshapeDelegate TensorReshape { get; init; }
        public required TensorCopyFromCpuFloatDelegate TensorCopyFromCpuFloat { get; init; }
        public required TensorCopyToCpuFloatDelegate TensorCopyToCpuFloat { get; init; }
        public required TensorGetShapeDelegate TensorGetShape { get; init; }
        public required TensorDestroyDelegate TensorDestroy { get; init; }
        public required OneDimArrayCstrDestroyDelegate OneDimArrayCstrDestroy { get; init; }
        public required OneDimArrayInt32DestroyDelegate OneDimArrayInt32Destroy { get; init; }

        public static Api Load(IntPtr library)
        {
            return new Api
            {
                ConfigCreate = GetDelegate<ConfigCreateDelegate>(library, "PD_ConfigCreate"),
                ConfigDestroy = GetDelegate<ConfigDestroyDelegate>(library, "PD_ConfigDestroy"),
                ConfigSetModel = GetDelegate<ConfigSetModelDelegate>(library, "PD_ConfigSetModel"),
                ConfigDisableGpu = GetDelegate<ConfigDisableGpuDelegate>(library, "PD_ConfigDisableGpu"),
                PredictorCreate = GetDelegate<PredictorCreateDelegate>(library, "PD_PredictorCreate"),
                PredictorDestroy = GetDelegate<PredictorDestroyDelegate>(library, "PD_PredictorDestroy"),
                PredictorGetInputNames = GetDelegate<PredictorGetInputNamesDelegate>(library, "PD_PredictorGetInputNames"),
                PredictorGetOutputNames = GetDelegate<PredictorGetOutputNamesDelegate>(library, "PD_PredictorGetOutputNames"),
                PredictorGetInputHandle = GetDelegate<PredictorGetInputHandleDelegate>(library, "PD_PredictorGetInputHandle"),
                PredictorGetOutputHandle = GetDelegate<PredictorGetOutputHandleDelegate>(library, "PD_PredictorGetOutputHandle"),
                PredictorRun = GetDelegate<PredictorRunDelegate>(library, "PD_PredictorRun"),
                TensorReshape = GetDelegate<TensorReshapeDelegate>(library, "PD_TensorReshape"),
                TensorCopyFromCpuFloat = GetDelegate<TensorCopyFromCpuFloatDelegate>(library, "PD_TensorCopyFromCpuFloat"),
                TensorCopyToCpuFloat = GetDelegate<TensorCopyToCpuFloatDelegate>(library, "PD_TensorCopyToCpuFloat"),
                TensorGetShape = GetDelegate<TensorGetShapeDelegate>(library, "PD_TensorGetShape"),
                TensorDestroy = GetDelegate<TensorDestroyDelegate>(library, "PD_TensorDestroy"),
                OneDimArrayCstrDestroy = GetDelegate<OneDimArrayCstrDestroyDelegate>(library, "PD_OneDimArrayCstrDestroy"),
                OneDimArrayInt32Destroy = GetDelegate<OneDimArrayInt32DestroyDelegate>(library, "PD_OneDimArrayInt32Destroy")
            };
        }

        private static T GetDelegate<T>(IntPtr library, string symbol) where T : Delegate
        {
            if (!NativeLibrary.TryGetExport(library, symbol, out var ptr))
            {
                throw new EntryPointNotFoundException($"Paddle C API symbol not found: {symbol}");
            }

            return Marshal.GetDelegateForFunctionPointer<T>(ptr);
        }

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate IntPtr ConfigCreateDelegate();

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void ConfigDestroyDelegate(IntPtr config);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void ConfigSetModelDelegate(
            IntPtr config,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string progFile,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string paramsFile);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void ConfigDisableGpuDelegate(IntPtr config);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate IntPtr PredictorCreateDelegate(IntPtr config);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void PredictorDestroyDelegate(IntPtr predictor);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate IntPtr PredictorGetInputNamesDelegate(IntPtr predictor);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate IntPtr PredictorGetOutputNamesDelegate(IntPtr predictor);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate IntPtr PredictorGetInputHandleDelegate(
            IntPtr predictor,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string name);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate IntPtr PredictorGetOutputHandleDelegate(
            IntPtr predictor,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string name);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate byte PredictorRunDelegate(IntPtr predictor);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void TensorReshapeDelegate(IntPtr tensor, nuint shapeSize, int[] shape);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void TensorCopyFromCpuFloatDelegate(IntPtr tensor, float[] data);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void TensorCopyToCpuFloatDelegate(IntPtr tensor, [Out] float[] data);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate IntPtr TensorGetShapeDelegate(IntPtr tensor);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void TensorDestroyDelegate(IntPtr tensor);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void OneDimArrayCstrDestroyDelegate(IntPtr array);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void OneDimArrayInt32DestroyDelegate(IntPtr array);
    }
}
