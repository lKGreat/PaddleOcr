using PaddleOcr.Models;

namespace PaddleOcr.Inference.Rec.Preprocessors;

/// <summary>
/// Rec 预处理器工厂，根据 RecAlgorithm 返回对应的预处理器实例。
/// </summary>
public static class RecPreprocessorFactory
{
    /// <summary>
    /// 根据算法创建对应的预处理器。
    /// </summary>
    public static IRecPreprocessor Create(RecAlgorithm algorithm, bool imageInverse = false)
    {
        return algorithm switch
        {
            RecAlgorithm.NRTR or RecAlgorithm.ViTSTR => new NrtrRecPreprocessor(),
            RecAlgorithm.SAR or RecAlgorithm.RobustScanner => new SarRecPreprocessor(),
            RecAlgorithm.SRN => new SrnRecPreprocessor(),
            RecAlgorithm.CAN => new CanRecPreprocessor(imageInverse),
            RecAlgorithm.LaTeXOCR or RecAlgorithm.PPFormulaNet_S or RecAlgorithm.PPFormulaNet_L
                or RecAlgorithm.PPFormulaNet_Plus_S or RecAlgorithm.PPFormulaNet_Plus_M
                or RecAlgorithm.PPFormulaNet_Plus_L => new LaTeXOcrPreprocessor(),
            RecAlgorithm.UniMERNet => new LaTeXOcrPreprocessor(),
            RecAlgorithm.SPIN => new SpinRecPreprocessor(),
            RecAlgorithm.VisionLAN => new VisionLanPreprocessor(),
            RecAlgorithm.RFL => new NrtrRecPreprocessor(),
            RecAlgorithm.ABINet => new DefaultRecPreprocessor(),
            RecAlgorithm.ParseQ or RecAlgorithm.CPPD or RecAlgorithm.CPPDPadding => new DefaultRecPreprocessor(),
            RecAlgorithm.SATRN => new SarRecPreprocessor(),
            RecAlgorithm.PREN => new DefaultRecPreprocessor(),
            _ => new DefaultRecPreprocessor()
        };
    }
}
