namespace PaddleOcr.Core.Errors;

public sealed class PocrException : Exception
{
    public PocrException(string message) : base(message)
    {
    }
}

