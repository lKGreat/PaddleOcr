using PaddleOcr.Core.Errors;

namespace PaddleOcr.Tools;

public sealed record ParsedCommand(
    string Root,
    string? Sub,
    string? ConfigPath,
    IReadOnlyList<string> Overrides,
    IReadOnlyDictionary<string, string> Options,
    IReadOnlyList<string> Positionals,
    IReadOnlyList<string> RawArgs);

public static class CommandLine
{
    public static ParsedCommand Parse(string[] args)
    {
        if (args.Length == 0)
        {
            throw new PocrException(GetHelp());
        }

        var root = args[0].Trim();
        string? sub = null;
        var cursor = 1;

        if (NeedsSubcommand(root))
        {
            if (args.Length < 2)
            {
                throw new PocrException($"Missing subcommand for '{root}'.\n{GetHelp()}");
            }

            sub = args[1].Trim();
            cursor = 2;
        }

        string? configPath = null;
        var overrides = new List<string>();
        var options = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        var positionals = new List<string>();

        while (cursor < args.Length)
        {
            var token = args[cursor];
            if (token is "-c" or "--config")
            {
                cursor++;
                if (cursor >= args.Length)
                {
                    throw new PocrException("Missing value for -c/--config");
                }

                configPath = args[cursor];
            }
            else if (token is "-o" or "--opt")
            {
                cursor++;
                while (cursor < args.Length && !IsFlag(args[cursor]))
                {
                    overrides.Add(args[cursor]);
                    cursor++;
                }

                cursor--;
            }
            else if (IsFlag(token))
            {
                var next = cursor + 1 < args.Length ? args[cursor + 1] : null;
                if (next is null || IsFlag(next))
                {
                    options[token] = "true";
                }
                else
                {
                    options[token] = next;
                    cursor++;
                }
            }
            else
            {
                positionals.Add(token);
            }

            cursor++;
        }

        return new ParsedCommand(root, sub, configPath, overrides, options, positionals, args);
    }

    public static string GetHelp()
    {
        return """
               pocr commands:
                 train -c <config> [-o K=V ...]
                 eval -c <config> [-o K=V ...]
                 export -c <config> [-o K=V ...]
                 export-onnx -c <config> [-o K=V ...]
                 export-center -c <config> [-o K=V ...]
                 infer <det|rec|cls|e2e|kie|kie-ser|kie-re|table|sr|system> [-c <config>] [options]
                 convert json2pdmodel --json_model_dir <dir> --output_dir <dir> --config <yml>
                 convert check-json-model --json_model_dir <dir>
                 config check -c <config>
                 config diff --base <path> --target <path>
                 doctor check-models [-c <config>] [--det_model_dir <path>] [--rec_model_dir <path>] [--cls_model_dir <path>] [--table_model_dir <path>] [--sr_model_dir <path>] [--kie_model_dir <path>]
                 doctor parity-table-kie -c <config> [--mode all|table|kie]
                 service test --server_url <url> --image_dir <dir> [--parallel N --timeout_ms MS --retries N --stress_rounds N --dump_failures true]
                 e2e <convert-label|eval> [args]
                 benchmark run --scenario <infer:system|service:test|e2e:eval|export:export-onnx|train:train> [--profile smoke|balanced|stress] [--warmup N --iterations N --continue_on_error true --report_json <path>] [scenario options...]
                 plugin <validate-package|verify-trust|load-runtime|load-runtime-dir> [--package_dir <dir>] [--plugins_root <dir>] [--require_trust true] [--allow_untrusted true]
               """;
    }

    private static bool NeedsSubcommand(string root)
    {
        return root.Equals("infer", StringComparison.OrdinalIgnoreCase)
            || root.Equals("convert", StringComparison.OrdinalIgnoreCase)
            || root.Equals("config", StringComparison.OrdinalIgnoreCase)
            || root.Equals("doctor", StringComparison.OrdinalIgnoreCase)
            || root.Equals("service", StringComparison.OrdinalIgnoreCase)
            || root.Equals("e2e", StringComparison.OrdinalIgnoreCase)
            || root.Equals("benchmark", StringComparison.OrdinalIgnoreCase)
            || root.Equals("plugin", StringComparison.OrdinalIgnoreCase);
    }

    private static bool IsFlag(string token) => token.StartsWith("-");
}
