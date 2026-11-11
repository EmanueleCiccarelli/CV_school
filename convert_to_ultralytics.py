"""
convert_to_ultralytics.py - Engine di conversione da BlockDL a Ultralytics YOLO

Questo modulo funge da engine di analisi, sanitizzazione e conversione:
1. AST Sanitizer: pulisce il codice generato da BlockDL (neutralizza print e valori fittizi di Linear).
2. Graph Parser (torch.fx): traccia il modello, rimuove funzioni finali incompatibili (log_softmax/softmax),
   gestisce il passaggio Grayscale -> RGB (se richiesto) e ricalcola dinamicamente le dimensioni dei layer.
3. Ultralytics Exporter: incapsula il modello in ClassificationModel e salva il checkpoint .pt (e architettura .yaml)
   pronto per il training con Ultralytics sia su Ubuntu sia su Windows.
"""

import ast
import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx


# =====================================================================
# 1. AST SANITIZER: Carica lo script BlockDL neutralizzando errori nei Linear
# =====================================================================
class _BlockDLASTSanitizer(ast.NodeTransformer):
    """
    Modifica l'AST dello script esportato da BlockDL prima dell'esecuzione:
    - Rimuove istanziazioni e print globali fuori dalle definizioni.
    - Sostituisce il primo argomento (in_features) di nn.Linear con 1, così gli studenti
      non devono calcolare la dimensione di flatten né rischiano crash per Out-Of-Memory.
    """

    def visit_Module(self, node: ast.Module) -> ast.Module:
        has_class_def = any(isinstance(stmt, ast.ClassDef) for stmt in node.body)
        cleaned_body = []
        for stmt in node.body:
            # Preserva import, classi e funzioni
            if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef)):
                cleaned_body.append(self.visit(stmt))
            # Se non ci sono classi (es. esportazione Sequential diretta), preserva le assegnazioni
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
            # Argomento posizionale: in_features
            if len(node.args) >= 1:
                node.args[0] = ast.Constant(value=1)
            # Argomento per parola chiave: in_features
            for kw in node.keywords:
                if kw.arg == "in_features":
                    kw.value = ast.Constant(value=1)
        return node


def load_blockdl_class(py_file_path: Path) -> nn.Module:
    """Legge il file .py di BlockDL ed esegue il caricamento sicuro del modello."""
    py_path = Path(py_file_path)
    if not py_path.exists():
        raise FileNotFoundError(f"File non trovato: {py_path}")

    source_code = py_path.read_text(encoding="utf-8")
    tree = ast.parse(source_code)
    sanitized_tree = _BlockDLASTSanitizer().visit(tree)
    ast.fix_missing_locations(sanitized_tree)

    namespace: Dict[str, Any] = {"torch": torch, "nn": nn, "F": F}
    exec(compile(sanitized_tree, filename=str(py_path), mode="exec"), namespace)

    # 1. Cerca una sottoclasse di nn.Module (pattern standard BlockDL)
    model_classes = [
        obj for obj in namespace.values()
        if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module
    ]
    if model_classes:
        return model_classes[-1]()

    # 2. Fallback: cerca un'istanza già inizializzata di nn.Module
    for obj in namespace.values():
        if isinstance(obj, nn.Module):
            return obj

    raise ValueError(f"Nessuna definizione o istanza di nn.Module trovata in {py_file_path}")


# =====================================================================
# 2. PARSER GRAFICO (torch.fx) + CALIBRAZIONE AUTOMATICA DIMENSIONI
# =====================================================================
FORBIDDEN_FINAL_FUNCS = {
    F.softmax,
    F.log_softmax,
    torch.softmax,
    getattr(torch, "log_softmax", None),
    F.sigmoid,
    torch.sigmoid,
}
FORBIDDEN_FINAL_MODULES = (nn.Softmax, nn.LogSoftmax, nn.Sigmoid)


def parse_and_calibrate_network(
    raw_model: nn.Module,
    imgsz: int = 28,
    force_rgb: bool = True,
    add_adaptive_pool: bool = True,
) -> Tuple[nn.Sequential, List[List], int, int]:
    """
    Analizza il grafo della rete, rimuove funzioni finali incompatibili (log_softmax/softmax),
    adatta i canali d'ingresso (RGB vs Grayscale) e ricalcola le dimensioni intermedie e finali.
    """
    traced: fx.GraphModule = fx.symbolic_trace(raw_model)
    modules_dict = dict(traced.named_modules())
    nodes = list(traced.graph.nodes)

    # 1. Identifica i nodi operativi
    op_nodes = [n for n in nodes if n.op not in ("placeholder", "output")]

    # 2. Rimuove attivazioni di probabilità finali (Ultralytics Classification usa CrossEntropyLoss sui logits)
    while op_nodes:
        last_node = op_nodes[-1]
        if last_node.op == "call_function" and last_node.target in FORBIDDEN_FINAL_FUNCS:
            func_name = getattr(last_node.target, "__name__", str(last_node.target))
            print(f"[Parser] Rimossa funzione finale incompatibile: {func_name}()")
            op_nodes.pop()
        elif last_node.op == "call_method" and last_node.target in ("softmax", "log_softmax", "sigmoid"):
            print(f"[Parser] Rimosso metodo finale incompatibile: .{last_node.target}()")
            op_nodes.pop()
        elif last_node.op == "call_module" and isinstance(modules_dict[last_node.target], FORBIDDEN_FINAL_MODULES):
            print(f"[Parser] Rimosso modulo finale incompatibile: {modules_dict[last_node.target]}")
            op_nodes.pop()
        else:
            break

    # 3. Estrae la sequenza ordinata di layer
    extracted_layers: List[nn.Module] = []
    for node in op_nodes:
        if node.op == "call_module":
            mod = copy.deepcopy(modules_dict[node.target])
            if isinstance(mod, nn.Sequential):
                extracted_layers.extend(list(mod.children()))
            else:
                extracted_layers.append(mod)

        elif node.op == "call_function":
            if node.target in (torch.flatten,):
                start_dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("start_dim", 1)
                extracted_layers.append(nn.Flatten(start_dim=start_dim))
            elif node.target in (F.relu, torch.relu):
                extracted_layers.append(nn.ReLU(inplace=True))
            elif node.target is F.leaky_relu:
                slope = node.kwargs.get("negative_slope", 0.01)
                extracted_layers.append(nn.LeakyReLU(negative_slope=slope, inplace=True))
            elif node.target is F.silu:
                extracted_layers.append(nn.SiLU(inplace=True))
            elif node.target is F.dropout:
                p = node.args[1] if len(node.args) > 1 else node.kwargs.get("p", 0.5)
                extracted_layers.append(nn.Dropout(p=p))
            elif node.target is F.max_pool2d:
                k = node.args[1] if len(node.args) > 1 else node.kwargs.get("kernel_size", 2)
                s = node.args[2] if len(node.args) > 2 else node.kwargs.get("stride", k)
                p = node.args[3] if len(node.args) > 3 else node.kwargs.get("padding", 0)
                extracted_layers.append(nn.MaxPool2d(kernel_size=k, stride=s, padding=p))
            elif node.target is F.avg_pool2d:
                k = node.args[1] if len(node.args) > 1 else node.kwargs.get("kernel_size", 2)
                s = node.args[2] if len(node.args) > 2 else node.kwargs.get("stride", k)
                p = node.args[3] if len(node.args) > 3 else node.kwargs.get("padding", 0)
                extracted_layers.append(nn.AvgPool2d(kernel_size=k, stride=s, padding=p))

        elif node.op == "call_method":
            if node.target in ("flatten", "view", "reshape"):
                extracted_layers.append(nn.Flatten(start_dim=1))
            elif node.target == "relu":
                extracted_layers.append(nn.ReLU(inplace=True))
            elif node.target == "silu":
                extracted_layers.append(nn.SiLU(inplace=True))

    # 4. Gestione Canali Input (RGB vs Grayscale) e calibrazione dinamica delle forme
    first_conv = next((m for m in extracted_layers if isinstance(m, nn.Conv2d)), None)
    if force_rgb:
        in_channels = 3
    else:
        in_channels = first_conv.in_channels if first_conv is not None else 3

    calibrated_layers: List[nn.Module] = []
    yaml_backbone: List[List] = []
    x = torch.zeros(1, in_channels, imgsz, imgsz)
    first_conv_handled = False

    for layer in extracted_layers:
        # A) Calibrazione Conv2d
        if isinstance(layer, nn.Conv2d):
            expected_in = x.shape[1]
            if not first_conv_handled:
                first_conv_handled = True
                if layer.in_channels != expected_in:
                    print(
                        f"[Parser] Primo Conv2d convertito da {layer.in_channels} "
                        f"a {expected_in} canali in ingresso."
                    )
            elif layer.in_channels != expected_in:
                print(
                    f"[Parser] Corretto mismatch Conv2d: in_channels {layer.in_channels} -> {expected_in}"
                )

            if layer.in_channels != expected_in:
                layer = nn.Conv2d(
                    in_channels=expected_in,
                    out_channels=layer.out_channels,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    dilation=layer.dilation,
                    groups=layer.groups if expected_in % layer.groups == 0 else 1,
                    bias=layer.bias is not None,
                    padding_mode=layer.padding_mode,
                )

        # B) Calibrazione BatchNorm
        elif isinstance(layer, nn.BatchNorm2d):
            if layer.num_features != x.shape[1]:
                print(f"[Parser] Corretto BatchNorm2d: {layer.num_features} -> {x.shape[1]} canali")
                layer = nn.BatchNorm2d(x.shape[1], eps=layer.eps, momentum=layer.momentum)

        elif isinstance(layer, nn.BatchNorm1d):
            if layer.num_features != x.shape[1]:
                layer = nn.BatchNorm1d(x.shape[1], eps=layer.eps, momentum=layer.momentum)

        # C) Inserimento AdaptiveAvgPool2d prima del Flatten
        elif isinstance(layer, nn.Flatten):
            already_pooled = bool(calibrated_layers and isinstance(calibrated_layers[-1], nn.AdaptiveAvgPool2d))
            if x.ndim == 4 and add_adaptive_pool and not already_pooled:
                h_feat, w_feat = x.shape[2], x.shape[3]
                pool_layer = nn.AdaptiveAvgPool2d((h_feat, w_feat))
                calibrated_layers.append(pool_layer)
                yaml_backbone.append([-1, 1, "nn.AdaptiveAvgPool2d", [[h_feat, w_feat]]])
                x = pool_layer(x)

        # D) Calibrazione automatica nn.Linear (nessun calcolo manuale agli studenti)
        elif isinstance(layer, nn.Linear):
            if x.ndim > 2:
                print("[Parser] Flatten mancante rilevato prima di Linear: inserito automaticamente.")
                already_pooled = bool(calibrated_layers and isinstance(calibrated_layers[-1], nn.AdaptiveAvgPool2d))
                if x.ndim == 4 and add_adaptive_pool and not already_pooled:
                    h_feat, w_feat = x.shape[2], x.shape[3]
                    pool_layer = nn.AdaptiveAvgPool2d((h_feat, w_feat))
                    calibrated_layers.append(pool_layer)
                    yaml_backbone.append([-1, 1, "nn.AdaptiveAvgPool2d", [[h_feat, w_feat]]])
                    x = pool_layer(x)
                flat_layer = nn.Flatten(start_dim=1)
                calibrated_layers.append(flat_layer)
                yaml_backbone.append([-1, 1, "nn.Flatten", [1]])
                x = flat_layer(x)

            actual_in_features = x.shape[1]
            if layer.in_features != actual_in_features:
                print(
                    f"[Parser] Dimensione Linear ricalcolata automaticamente: "
                    f"in_features={actual_in_features} (out_features={layer.out_features})"
                )
                layer = nn.Linear(
                    in_features=actual_in_features,
                    out_features=layer.out_features,
                    bias=layer.bias is not None,
                )

        x = layer(x)
        calibrated_layers.append(layer)
        yaml_backbone.append(_layer_to_ultralytics_yaml_row(layer))

    if x.ndim != 2:
        raise RuntimeError(
            f"L'uscita finale della rete non è un vettore 2D (Batch, Classi), ma ha forma {tuple(x.shape)}."
        )

    num_classes = int(x.shape[1])
    sequential_model = nn.Sequential(*calibrated_layers)
    return sequential_model, yaml_backbone, num_classes, in_channels


def _layer_to_ultralytics_yaml_row(layer: nn.Module) -> List:
    """Converte un layer PyTorch in riga serializzabile per il modello Ultralytics."""
    if isinstance(layer, nn.Conv2d):
        return [
            -1, 1, "nn.Conv2d",
            [
                layer.in_channels,
                layer.out_channels,
                list(layer.kernel_size),
                list(layer.stride),
                list(layer.padding) if isinstance(layer.padding, tuple) else layer.padding,
                list(layer.dilation),
                layer.groups,
                layer.bias is not None,
            ],
        ]
    if isinstance(layer, nn.MaxPool2d):
        k = list(layer.kernel_size) if isinstance(layer.kernel_size, tuple) else layer.kernel_size
        s = list(layer.stride) if isinstance(layer.stride, tuple) else layer.stride
        p = list(layer.padding) if isinstance(layer.padding, tuple) else layer.padding
        return [-1, 1, "nn.MaxPool2d", [k, s, p]]
    if isinstance(layer, nn.AvgPool2d):
        k = list(layer.kernel_size) if isinstance(layer.kernel_size, tuple) else layer.kernel_size
        s = list(layer.stride) if isinstance(layer.stride, tuple) else layer.stride
        p = list(layer.padding) if isinstance(layer.padding, tuple) else layer.padding
        return [-1, 1, "nn.AvgPool2d", [k, s, p]]
    if isinstance(layer, nn.AdaptiveAvgPool2d):
        out_sz = list(layer.output_size) if isinstance(layer.output_size, tuple) else layer.output_size
        return [-1, 1, "nn.AdaptiveAvgPool2d", [out_sz]]
    if isinstance(layer, nn.Flatten):
        return [-1, 1, "nn.Flatten", [layer.start_dim, layer.end_dim]]
    if isinstance(layer, nn.Linear):
        return [-1, 1, "nn.Linear", [layer.in_features, layer.out_features, layer.bias is not None]]
    if isinstance(layer, nn.BatchNorm2d):
        return [-1, 1, "nn.BatchNorm2d", [layer.num_features]]
    if isinstance(layer, nn.BatchNorm1d):
        return [-1, 1, "nn.BatchNorm1d", [layer.num_features]]
    if isinstance(layer, nn.ReLU):
        return [-1, 1, "nn.ReLU", [layer.inplace]]
    if isinstance(layer, nn.LeakyReLU):
        return [-1, 1, "nn.LeakyReLU", [layer.negative_slope, layer.inplace]]
    if isinstance(layer, nn.Dropout):
        return [-1, 1, "nn.Dropout", [layer.p]]
    if isinstance(layer, nn.SiLU):
        return [-1, 1, "nn.SiLU", [layer.inplace]]
    return [-1, 1, f"nn.{layer.__class__.__name__}", []]


# =====================================================================
# 3. ESPORTAZIONE IN FORMATO .PT (E .YAML) COMPATIBILE ULTRALYTICS
# =====================================================================
def export_to_ultralytics_pt(
    py_file_path: Path | str,
    output_pt_path: Path | str = "model.pt",
    imgsz: int = 28,
    force_rgb: bool = True,
    class_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """
    Pipeline completa di conversione:
    1. Legge il file .py generato da BlockDL.
    2. Adatta e calibra la rete (logits puri, RGB, calcolo automatico dimensioni Linear).
    3. Costruisce e salva il checkpoint .pt e l'architettura .yaml per Ultralytics Classification.

    Restituisce un dizionario con i metadati della conversione.
    """
    py_path = Path(py_file_path).resolve()
    out_pt = Path(output_pt_path).resolve()
    out_yaml = out_pt.with_suffix(".yaml")

    raw_model = load_blockdl_class(py_path)
    seq_model, yaml_rows, nc, ch = parse_and_calibrate_network(
        raw_model=raw_model,
        imgsz=imgsz,
        force_rgb=force_rgb,
        add_adaptive_pool=True,
    )

    if class_names is None:
        if nc == 2:
            class_names = {0: "1", 1: "7"}
        else:
            class_names = {i: f"classe_{i}" for i in range(nc)}

    # Struttura YAML per il parse_model di Ultralytics
    yaml_dict = {
        "nc": nc,
        "ch": ch,
        "scales": {"n": [1.0, 1.0, 1024]},
        "backbone": yaml_rows[:-1],
        "head": [yaml_rows[-1]],
    }

    # Salva il file .yaml di architettura
    saved_yaml = False
    try:
        import yaml
        with open(out_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_dict, f, sort_keys=False)
        saved_yaml = True
    except ImportError:
        try:
            from ultralytics.utils import YAML
            YAML.save(out_yaml, yaml_dict)
            saved_yaml = True
        except Exception:
            pass

    # Incapsula in ClassificationModel di Ultralytics
    try:
        from ultralytics.nn.tasks import ClassificationModel

        ul_model = ClassificationModel(cfg=yaml_dict, ch=ch, nc=nc, verbose=False)
        ul_model.model.load_state_dict(seq_model.state_dict(), strict=False)
        ul_model.names = class_names
        ul_model.yaml = yaml_dict
        ul_model.stride = torch.tensor([1])
        model_to_save = ul_model.eval()
        print("[Exporter] Modello incapsulato con successo in ultralytics.nn.tasks.ClassificationModel.")
    except ImportError:
        for i, m in enumerate(seq_model):
            m.i, m.f, m.type = i, -1, f"nn.{m.__class__.__name__}"
        seq_model.nc = nc
        seq_model.names = class_names
        seq_model.yaml = yaml_dict
        seq_model.stride = torch.tensor([1])
        seq_model.args = {"task": "classify", "imgsz": imgsz}
        model_to_save = seq_model.eval()
        print("[Exporter] Ultralytics non rilevato: salvato come nn.Sequential compatibile.")

    # Costruzione del checkpoint nativo Ultralytics
    ckpt = {
        "epoch": -1,
        "best_fitness": None,
        "model": model_to_save,
        "ema": None,
        "updates": 0,
        "optimizer": None,
        "train_args": {
            "task": "classify",
            "model": out_pt.name,
            "imgsz": imgsz,
        },
        "train_metrics": {},
        "date": datetime.now().isoformat(),
        "version": "8.3.0",
    }

    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, str(out_pt))

    print(f"\n[OK] Modello salvato con successo: {out_pt}")
    if saved_yaml:
        print(f"[OK] Architettura YAML salvata: {out_yaml}")
    print(f"     - Canali input: {ch} | Risoluzione base: {imgsz}x{imgsz} | Classi: {nc} {class_names}")

    return {
        "output_pt_path": str(out_pt),
        "output_yaml_path": str(out_yaml) if saved_yaml else None,
        "num_classes": nc,
        "in_channels": ch,
        "imgsz": imgsz,
        "class_names": class_names,
    }


if __name__ == "__main__":
    # Test diretto sul file di default se eseguito come script
    default_input = Path(__file__).resolve().parent / "network to convert" / "pytorch_model_sequential (1).py"
    if default_input.exists():
        export_to_ultralytics_pt(
            py_file_path=default_input,
            output_pt_path="model.pt",
            imgsz=28,
            force_rgb=True,
            class_names={0: "1", 1: "7"},
        )
    else:
        print(f"File di test non trovato in: {default_input}")
