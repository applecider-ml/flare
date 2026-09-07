"""Export the bts6 boosters to ONNX for brokers (BOOM keeps every model as ONNX in Git LFS).

Why not onnxmltools: it stores split thresholds in float32 and applies a softmax,
which moved 0.8% of raw scores by up to 0.6 on the benchmark and discarded the
normaliser the anomaly energy needs. Here every tree is written as a
TreeEnsembleRegressor with double thresholds (`nodes_values_as_tensor`), raw SUM
aggregation and no post-transform, so the output equals `Booster.predict(raw_score=True)`
to float32 output precision (max 2.4e-7 on 3000 benchmark rows). Softmax, sigmoid,
the SLSN branch, conformal sets and the energy are computed downstream in double.

Each file carries what a consumer needs as ONNX metadata_props, so the .onnx files
are the complete artefact: `feature_names` (JSON list, the input column order),
`objective`, `num_class`, `flare_model`, `flare_version`; the top model also carries
`classes`, `top_classes`, `branch_threshold`, `conformal` (conformal.json),
`anomaly_calibration` (anomaly_calibration.json) and `model_card`.

    python scripts/export_onnx.py [--model-dir models/bts6] [--out models/bts6/onnx]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
VERSION = "2.0.0"


def lgbm_to_onnx(booster: lgb.Booster, name: str, extra_meta: dict) -> onnx.ModelProto:
    d = booster.dump_model()
    ntpi = d.get("num_tree_per_iteration", 1)
    nodes = dict(treeids=[], nodeids=[], featureids=[], values=[], modes=[], truenodeids=[], falsenodeids=[], missing=[])
    targets = dict(treeids=[], nodeids=[], ids=[], weights=[])
    for ti, tree in enumerate(d["tree_info"]):
        target = ti % ntpi
        counter = [0]

        def walk(node):
            nid = counter[0]; counter[0] += 1
            if "split_feature" not in node:                      # leaf (also a single-leaf tree)
                for k, v in (("treeids", ti), ("nodeids", nid), ("featureids", 0), ("values", 0.0),
                             ("modes", "LEAF"), ("truenodeids", 0), ("falsenodeids", 0), ("missing", 0)):
                    nodes[k].append(v)
                targets["treeids"].append(ti); targets["nodeids"].append(nid)
                targets["ids"].append(target); targets["weights"].append(float(node["leaf_value"]))
                return nid
            if node["decision_type"] != "<=":
                raise ValueError(f"unsupported decision type {node['decision_type']} (categorical split?)")
            thr, mt, dl = float(node["threshold"]), node["missing_type"], bool(node["default_left"])
            # LightGBM semantics: NaN with missing_type None is treated as 0.0 and compared;
            # NaN with missing_type NaN follows default_left. missing_type Zero also redirects
            # genuine zeros, which ONNX cannot express -> refuse.
            if mt == "None":
                miss_true = 1 if 0.0 <= thr else 0
            elif mt == "NaN":
                miss_true = 1 if dl else 0
            else:
                raise ValueError("missing_type Zero is not representable in ONNX TreeEnsemble")
            nodes["treeids"].append(ti); nodes["nodeids"].append(nid)
            nodes["featureids"].append(int(node["split_feature"])); nodes["values"].append(thr)
            nodes["modes"].append("BRANCH_LEQ"); nodes["missing"].append(miss_true)
            nodes["truenodeids"].append(-1); nodes["falsenodeids"].append(-1); slot = len(nodes["truenodeids"]) - 1
            left, right = walk(node["left_child"]), walk(node["right_child"])
            nodes["truenodeids"][slot], nodes["falsenodeids"][slot] = left, right
            return nid

        walk(tree["tree_structure"])
    nfeat = len(booster.feature_name())
    node = helper.make_node(
        "TreeEnsembleRegressor", ["X"], ["raw"], domain="ai.onnx.ml",
        n_targets=ntpi, aggregate_function="SUM", post_transform="NONE",
        nodes_treeids=nodes["treeids"], nodes_nodeids=nodes["nodeids"], nodes_featureids=nodes["featureids"],
        nodes_values_as_tensor=numpy_helper.from_array(np.array(nodes["values"], dtype=np.float64), "nodes_values"),
        nodes_modes=nodes["modes"], nodes_truenodeids=nodes["truenodeids"], nodes_falsenodeids=nodes["falsenodeids"],
        nodes_missing_value_tracks_true=nodes["missing"],
        target_treeids=targets["treeids"], target_nodeids=targets["nodeids"], target_ids=targets["ids"],
        target_weights_as_tensor=numpy_helper.from_array(np.array(targets["weights"], dtype=np.float64), "target_weights"),
    )
    graph = helper.make_graph(
        [node], name,
        [helper.make_tensor_value_info("X", TensorProto.DOUBLE, [None, nfeat])],
        [helper.make_tensor_value_info("raw", TensorProto.FLOAT, [None, ntpi])],
        doc_string="FLARE LightGBM booster as raw-score TreeEnsembleRegressor (double thresholds)")
    model = helper.make_model(graph, producer_name="flare", producer_version=VERSION,
                              opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("ai.onnx.ml", 3)])
    model.ir_version = 8
    meta = dict(feature_names=json.dumps(booster.feature_name()), objective=d.get("objective", ""),
                num_class=str(ntpi), flare_model=name, flare_version=VERSION)
    meta.update(extra_meta)
    for k, v in meta.items():
        p = model.metadata_props.add(); p.key, p.value = k, v
    onnx.checker.check_model(model)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/bts6"); ap.add_argument("--out", default="models/bts6/onnx")
    ap.add_argument("--check-rows", type=int, default=3000)
    a = ap.parse_args()
    md, out = Path(a.model_dir), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    card = json.loads((md / "model_card.json").read_text())
    top_meta = dict(classes=json.dumps(card["classes"]), top_classes=json.dumps(card["top_classes"]),
                    branch_threshold=str(card["branch_threshold"]),
                    conformal=(md / "conformal.json").read_text(),
                    anomaly_calibration=(md / "anomaly_calibration.json").read_text(),
                    model_card=(md / "model_card.json").read_text())
    boosters = {"top": (lgb.Booster(model_file=str(md / "top.txt")), top_meta),
                "slsn_branch": (lgb.Booster(model_file=str(md / "slsn_branch.txt")), {}),
                "ad_space": (lgb.Booster(model_file=str(md / "ad_space.txt")), {})}
    import onnxruntime as ort, pandas as pd
    from flare.hierarchical import HierarchicalFlare
    clf = HierarchicalFlare.from_pretrained(); FN = list(clf.feature_names)
    X = pd.read_parquet("benchmark/features_all.parquet")
    for f in ["hosts.csv", "hosts_photoz.csv", "context_gaia_wise.csv", "ps1_position.csv"]:
        dd = pd.read_csv("benchmark/" + f).drop_duplicates("obj_id")
        X = X.merge(dd[["obj_id"] + [c for c in dd.columns if c in FN and c not in X.columns]], on="obj_id", how="left")
    for c in FN:
        if c not in X.columns: X[c] = np.nan
    X = X.sample(a.check_rows, random_state=0)
    for name, (b, meta) in boosters.items():
        m = lgbm_to_onnx(b, name, meta); path = out / f"flare_bts6_{name}.onnx"; onnx.save(m, path)
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        got = sess.run(None, {"X": X[b.feature_name()].values.astype(np.float64)})[0]
        ref = b.predict(X[b.feature_name()], raw_score=True).reshape(got.shape)
        dmax = float(np.abs(got - ref).max())
        print(f"{path}  {path.stat().st_size/1e6:.1f} MB  raw-score max|diff| vs LightGBM {dmax:.1e}  metadata {sorted(meta)+['feature_names','objective','num_class','flare_model','flare_version']}")
        assert dmax < 1e-5, "ONNX export is not faithful"
    print("done")


if __name__ == "__main__":
    main()
