using PaddleOcr.Models;

namespace PaddleOcr.Inference.Rec.Postprocessors;

/// <summary>
/// Rec 后处理解码器工厂，根据算法名称或 RecAlgorithm 创建对应的解码器。
/// </summary>
public static class RecPostprocessorFactory
{
    /// <summary>
    /// 根据后处理器名称创建解码器实例。
    /// </summary>
    public static IRecPostprocessor Create(string name)
    {
        return name.ToLowerInvariant() switch
        {
            "ctc" or "ctc-greedy" or "ctclabeldecode" => new CtcLabelDecoder(),
            "attn" or "attention" or "attnlabeldecode" => new AttnLabelDecoder(),
            "srn" or "srnlabeldecode" => new SrnLabelDecoder(),
            "nrtr" or "ntrlabeldecode" => new NrtrLabelDecoder(),
            "sar" or "sarlabeldecode" => new SarLabelDecoder(),
            "vitstr" or "vitstrlabeldecode" => new ViTStrLabelDecoder(),
            "abinet" or "abinetlabeldecode" => new ABINetLabelDecoder(),
            "spin" or "spinlabeldecode" => new SpinLabelDecoder(),
            "can" or "canlabeldecode" => new CanLabelDecoder(),
            "latexocr" or "latexocrdecode" => new LaTeXOcrDecoder(),
            "parseq" or "parseqlabeldecode" => new ParseQLabelDecoder(),
            "cppd" or "cppdlabeldecode" => new CppdLabelDecoder(),
            "pren" or "prenlabeldecode" => new PrenLabelDecoder(),
            "unimernet" or "unimernet_decode" or "unimernetdecode" => new UniMerNetDecoder(),
            "visionlan" or "vllabeldecode" => new VisionLanDecoder(),
            "rfl" or "rfllabeldecode" => new RflLabelDecoder(),
            "satrn" or "satrnlabeldecode" => new SatrnLabelDecoder(),
            _ => new CtcLabelDecoder()
        };
    }

    /// <summary>
    /// 根据 RecAlgorithm 创建对应的默认解码器。
    /// </summary>
    public static IRecPostprocessor Create(RecAlgorithm algorithm)
    {
        return Create(algorithm.GetDefaultPostprocessor());
    }
}
