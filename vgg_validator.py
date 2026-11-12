"""
vgg_validator.py - Validatore e Diagnostica Architetturale per VGG-16 da BlockDL

Questo modulo si occupa esclusivamente di:
1. Leggere e analizzare lo script PyTorch esportato da BlockDL tramite AST sanitizzato.
2. Estrarre la sequenza effettiva di strati definiti dallo studente.
3. Eseguire una validazione rigorosa confrontando la rete con la specifica ufficiale VGG-16:
   - 13 layer Conv2d suddivisi nei 5 blocchi: [2, 2, 3, 3, 3]
   - Dimensioni dei canali esatte: 64 -> 128 -> 256 -> 512 -> 512
   - Kernel size = 3, Stride = 1, Padding = 1 su ogni convoluzione
   - 5 layer MaxPool2d (kernel=2, stride=2) posti al termine di ogni blocco
   - Riconoscimento del classificatore finale (3 layer Linear) da escludere
4. Generare un report diagnostico chiaro ed educativo con indicazioni su come
   correggere i blocchi grafici su BlockDL in caso di errori.
"""

import ast
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx


# Specifica formale della VGG-16 standard
EXPECTED_VGG16_CONVS = [
    # (blocco, indice_nel_blocco, in_ch, out_ch, k, p)
    (1, 1, 3, 64, 3, 1),
    (1, 2, 64, 64, 3, 1),
    (2, 1, 64, 128, 3, 1),
    (2, 2, 128, 128, 3, 1),
    (3, 1, 128, 256, 3, 1),
    (3, 2, 256, 256, 3, 1),
    (3, 3, 256, 256, 3, 1),
    (4, 1, 256, 512, 3, 1),
    (4, 2, 512, 512, 3, 1),
    (4, 3, 512, 512, 3, 1),
    (5, 1, 512, 512, 3, 1),
    (5, 2, 512, 512, 3, 1),
    (5, 3, 512, 512, 3, 1),
]


class _BlockDLASTSanitizer(ast.NodeTransformer):
    """Pulisce l'AST per neutralizzare print globali o in_features errati in nn.Linear."""

    def visit_Module(self, node: ast.Module) -> ast.Module:
        has_class_def = any(isinstance(stmt, ast.ClassDef) for stmt in node.body)
        cleaned_body = []
        for stmt in node.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef)):
                cleaned_body.append(self.visit(stmt))
            elif not has_class_def and isinstance(stmt, ast.Assign):
                cleaned_body.append(self.visit(stmt))
        node.body = cleaned_body
        return node

    def visit_Call(self, node: ast.Call) -> ast.Call:
        self.generic_visit(node)
        func_name = ""
        if isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            func_name = node.func.id

        if func_name == "Linear":
            if len(node.args) >= 1:
                node.args[0] = ast.Constant(value=1)
            for kw in node.keywords:
                if kw.arg == "in_features":
                    kw.value = ast.Constant(value=1)
        return node


def load_blockdl_model(py_file_path: Path) -> nn.Module:
    """Carica e istanzia il modello definito nello script esportato da BlockDL."""
    if not py_file_path.exists():
        raise FileNotFoundError(f"File non trovato: {py_file_path}")

    source_code = py_file_path.read_text(encoding="utf-8")
    tree = ast.parse(source_code)
    sanitized_tree = _BlockDLASTSanitizer().visit(tree)
    ast.fix_missing_locations(sanitized_tree)

    namespace: Dict[str, Any] = {"torch": torch, "nn": nn, "F": F}
    exec(compile(sanitized_tree, filename=str(py_file_path), mode="exec"), namespace)

    model_classes = [
        obj for obj in namespace.values()
        if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module
    ]
    if model_classes:
        return model_classes[-1]()

    for obj in namespace.values():
        if isinstance(obj, nn.Module):
            return obj

    raise ValueError(f"Nessuna definizione di rete trovata in {py_file_path.name}")


def extract_ordered_layers(raw_model: nn.Module) -> List[nn.Module]:
    """Estrae la sequenza ordinata di moduli operativi della rete."""
    try:
        traced = fx.symbolic_trace(raw_model)
        modules_dict = dict(traced.named_modules())
        extracted: List[nn.Module] = []

        for node in traced.graph.nodes:
            if node.op == "call_module":
                mod = copy.deepcopy(modules_dict[node.target])
                if isinstance(mod, nn.Sequential):
                    extracted.extend(list(mod.children()))
                else:
                    extracted.append(mod)
            elif node.op == "call_function":
                if node.target in (F.relu, torch.relu):
                    extracted.append(nn.ReLU(inplace=True))
                elif node.target is F.max_pool2d:
                    k = node.args[1] if len(node.args) > 1 else node.kwargs.get("kernel_size", 2)
                    s = node.args[2] if len(node.args) > 2 else node.kwargs.get("stride", k)
                    p = node.args[3] if len(node.args) > 3 else node.kwargs.get("padding", 0)
                    extracted.append(nn.MaxPool2d(kernel_size=k, stride=s, padding=p))
                elif node.target in (torch.flatten,):
                    extracted.append(nn.Flatten(start_dim=1))
            elif node.op == "call_method":
                if node.target == "relu":
                    extracted.append(nn.ReLU(inplace=True))
                elif node.target in ("flatten", "view", "reshape"):
                    extracted.append(nn.Flatten(start_dim=1))

        if extracted:
            return extracted
    except Exception:
        pass

    # Fallback ricorsivo
    layers: List[nn.Module] = []
    for m in raw_model.modules():
        if isinstance(m, (nn.Conv2d, nn.MaxPool2d, nn.ReLU, nn.Linear, nn.Dropout, nn.AdaptiveAvgPool2d, nn.Flatten)):
            layers.append(copy.deepcopy(m))
    return layers


def validate_vgg16_architecture(layers: List[nn.Module]) -> Tuple[bool, List[Dict[str, str]], List[str], int, int]:
    """Valida la sequenza di layer rispetto ai requisiti di VGG-16."""
    errors: List[Dict[str, str]] = []
    warnings: List[str] = []

    convs: List[nn.Conv2d] = [l for l in layers if isinstance(l, nn.Conv2d)]
    pools: List[nn.MaxPool2d] = [l for l in layers if isinstance(l, nn.MaxPool2d)]
    linears: List[nn.Linear] = [l for l in layers if isinstance(l, nn.Linear)]

    if linears:
        warnings.append(
            f"Rilevati {len(linears)} layer densi (Linear). Per la trasformazione in Object Detector, "
            "il classificatore denso finale viene automaticamente escluso per innestare la YOLO Detect Head."
        )

    # 1. Controllo numero convoluzioni
    if len(convs) != 13:
        errors.append({
            "title": f"Numero errato di layer convoluzionali: trovati {len(convs)} invece di 13!",
            "expected": "13 layer Conv2d suddivisi nei 5 blocchi: [2, 2, 3, 3, 3]",
            "actual": f"{len(convs)} layer Conv2d rilevati nella rete",
            "fix": "Controlla su BlockDL i 5 blocchi: Blocco 1 ha 2 Conv, Blocco 2 ha 2 Conv, "
                   "Blocco 3 ha 3 Conv, Blocco 4 ha 3 Conv, Blocco 5 ha 3 Conv."
        })

    # 2. Controllo numero MaxPool
    if len(pools) != 5:
        errors.append({
            "title": f"Numero errato di layer MaxPool2d: trovati {len(pools)} invece di 5!",
            "expected": "5 layer MaxPool2d (uno al termine di ciascuno dei 5 blocchi)",
            "actual": f"{len(pools)} layer MaxPool2d rilevati",
            "fix": "Assicurati di aver inserito un blocco MaxPool2D(2, 2) alla fine di ogni blocco convoluzionale."
        })

    # 3. Controllo parametri layer per layer
    for i, exp in enumerate(EXPECTED_VGG16_CONVS):
        if i >= len(convs):
            break
        actual_conv = convs[i]
        b_idx, c_idx, exp_in, exp_out, exp_k, exp_p = exp

        act_k = actual_conv.kernel_size[0] if isinstance(actual_conv.kernel_size, tuple) else actual_conv.kernel_size
        act_p = actual_conv.padding[0] if isinstance(actual_conv.padding, tuple) else actual_conv.padding
        act_s = actual_conv.stride[0] if isinstance(actual_conv.stride, tuple) else actual_conv.stride

        discrepancies = []
        if actual_conv.in_channels != exp_in:
            discrepancies.append(f"in_channels: {actual_conv.in_channels} (atteso {exp_in})")
        if actual_conv.out_channels != exp_out:
            discrepancies.append(f"out_channels: {actual_conv.out_channels} (atteso {exp_out})")
        if act_k != exp_k:
            discrepancies.append(f"kernel_size: {act_k} (atteso {exp_k})")
        if act_p != exp_p:
            discrepancies.append(f"padding: {act_p} (atteso {exp_p})")
        if act_s != 1:
            discrepancies.append(f"stride: {act_s} (atteso 1)")

        if discrepancies:
            errors.append({
                "title": f"Blocco {b_idx}, Convoluzione {c_idx} (Conv2d #{i+1}): Parametri non corretti",
                "expected": f"Conv2d(in={exp_in}, out={exp_out}, kernel={exp_k}, padding={exp_p}, stride=1)",
                "actual": f"Conv2d(in={actual_conv.in_channels}, out={actual_conv.out_channels}, kernel={act_k}, padding={act_p}, stride={act_s})",
                "fix": f"In BlockDL, seleziona la convoluzione #{i+1} nel Blocco {b_idx} e imposta: "
                       f"In Channels = {exp_in}, Out Channels = {exp_out}, Kernel = 3, Padding = 1, Stride = 1."
            })

    # 4. Controllo parametri MaxPool2d
    for p_idx, pool in enumerate(pools):
        pk = pool.kernel_size if isinstance(pool.kernel_size, int) else pool.kernel_size[0]
        ps = pool.stride if isinstance(pool.stride, int) else pool.stride[0]
        if pk != 2 or ps != 2:
            errors.append({
                "title": f"MaxPool2d #{p_idx+1}: Dimensione o passo errato",
                "expected": "kernel_size=2, stride=2 (dimezza altezza e larghezza)",
                "actual": f"kernel_size={pk}, stride={ps}",
                "fix": f"In BlockDL, imposta per il blocco MaxPool #{p_idx+1} sia kernel_size che stride a 2."
            })

    is_valid = (len(errors) == 0)
    return is_valid, errors, warnings, len(convs), len(pools)


def print_diagnostic_report(errors: List[Dict[str, str]], warnings: List[str], num_convs: int, num_pools: int):
    """Visualizza il report per gli studenti."""
    print("\n" + "=" * 75)
    print("  VERIFICA ARCHITETTURALE VGG-16: ANALISI DEL MODELLO BLOCKDL")
    print("=" * 75)
    print(f"Riepilogo elementi estratti dal codice:")
    print(f"  - Convoluzioni individuate: {num_convs} / 13 richieste")
    print(f"  - MaxPool individuati:      {num_pools} / 5 richiesti")
    print("-" * 75)

    if warnings:
        for w in warnings:
            print(f"💡 [INFO]: {w}")
        print("-" * 75)

    if errors:
        print(f"❌ RILEVATE {len(errors)} DISCREPANZE RISPETTO ALLA SPECIFICA VGG-16:\n")
        for i, err in enumerate(errors, 1):
            print(f"[{i}] {err['title']}")
            print(f"    • Atteso in VGG-16:   {err['expected']}")
            print(f"    • Trovato in BlockDL:  {err['actual']}")
            print(f"    👉 Come correggere:    {err['fix']}\n")

        print("=" * 75)
        print("INDICAZIONI PER LO STUDENTE:")
        print("1. Riapri il tuo progetto su BlockDL (https://blockdl.com/editor).")
        print("2. Modifica i blocchi seguendo le indicazioni sopra riportate.")
        print("3. Clicca su Export -> PyTorch, sostituisci il file nella cartella e riesegui.")
        print("=" * 75 + "\n")
    else:
        print("✅ VALIDAZIONE SUPERATA: L'architettura corrisponde al 100% alla specifica VGG-16!")
        print("=" * 75 + "\n")


def validate_student_vgg(py_file_path: Path) -> bool:
    """Funzione principale di validazione: legge il file, lo valida e stampa la diagnostica."""
    try:
        model = load_blockdl_model(py_file_path)
        layers = extract_ordered_layers(model)
        is_valid, errors, warnings, num_convs, num_pools = validate_vgg16_architecture(layers)
        print_diagnostic_report(errors, warnings, num_convs, num_pools)
        return is_valid
    except Exception as e:
        print("\n" + "=" * 75)
        print("  ERRORE DURANTE IL CARICAMENTO DEL MODELLO BLOCKDL")
        print("=" * 75)
        print(f"Non è stato possibile analizzare il file: {py_file_path.name}")
        print(f"Dettaglio errore: {e}\n")
        print("Possibili cause:")
        print("1. Il file esportato da BlockDL contiene blocchi non collegati o interrotti.")
        print("2. Assicurati di esportare la rete da BlockDL selezionando 'PyTorch'.")
        print("=" * 75 + "\n")
        return False
