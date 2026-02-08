#!/usr/bin/env dotnet-script

// Quick script to test MultiHead config extraction

using System;
using System.Collections.Generic;
using System.Linq;

// Simulate YAML config parsing
var headList = new List<object?>
{
    new Dictionary<string, object?>
    {
        ["CTCHead"] = new Dictionary<string, object?>
        {
            ["Neck"] = new Dictionary<string, object?>
            {
                ["name"] = "svtr",
                ["dims"] = 120L,
                ["depth"] = 2L,
                ["hidden_dims"] = 120L,
                ["kernel_size"] = new List<object?> { 1L, 3L },
                ["use_guide"] = true
            },
            ["Head"] = new Dictionary<string, object?>
            {
                ["fc_decay"] = 0.00001
            }
        }
    },
    new Dictionary<string, object?>
    {
        ["NRTRHead"] = new Dictionary<string, object?>
        {
            ["nrtr_dim"] = 384L,
            ["max_text_length"] = 25L
        }
    }
};

// Test extraction logic
Console.WriteLine("=== Testing MultiHead Config Extraction ===");
Console.WriteLine($"head_list count: {headList.Count}");

// Extract CTC
if (headList[0] is Dictionary<string, object?> firstHead &&
    firstHead.TryGetValue("CTCHead", out var ctcCfgRaw) &&
    ctcCfgRaw is Dictionary<string, object?> ctcCfg &&
    ctcCfg.TryGetValue("Neck", out var neckCfgRaw) &&
    neckCfgRaw is Dictionary<string, object?> neckCfg)
{
    Console.WriteLine("✓ CTC Neck config found");
    var encoderType = neckCfg.TryGetValue("name", out var nameObj) ? nameObj?.ToString() ?? "reshape" : "reshape";
    var dims = neckCfg.TryGetValue("dims", out var dimsObj) ? ToInt(dimsObj) : 0;
    var depth = neckCfg.TryGetValue("depth", out var depthObj) ? ToInt(depthObj) : 1;
    var hiddenDims = neckCfg.TryGetValue("hidden_dims", out var hiddenObj) ? ToInt(hiddenObj) : 0;

    Console.WriteLine($"  encoder_type: {encoderType}");
    Console.WriteLine($"  dims: {dims}");
    Console.WriteLine($"  depth: {depth}");
    Console.WriteLine($"  hidden_dims: {hiddenDims}");

    if (dims > 0 && !string.Equals(encoderType, "reshape", StringComparison.OrdinalIgnoreCase))
    {
        Console.WriteLine("✓ CTC encoder config VALID (will be created)");
    }
    else
    {
        Console.WriteLine("✗ CTC encoder config INVALID (will be NULL)");
    }
}

// Extract NRTR
if (headList[1] is Dictionary<string, object?> secondHead &&
    secondHead.TryGetValue("NRTRHead", out var nrtrCfgRaw) &&
    nrtrCfgRaw is Dictionary<string, object?> nrtrCfg &&
    nrtrCfg.TryGetValue("nrtr_dim", out var dimObj))
{
    var nrtrDim = ToInt(dimObj);
    Console.WriteLine($"✓ NRTR dim: {nrtrDim}");
}

static int ToInt(object? obj)
{
    if (obj is null) return 0;
    return obj switch
    {
        int i => i,
        long l => (int)l,
        float f => (int)f,
        double d => (int)d,
        decimal m => (int)m,
        _ => int.TryParse(obj.ToString(), out var parsed) ? parsed : 0
    };
}
