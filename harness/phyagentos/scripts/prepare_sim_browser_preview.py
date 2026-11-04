#!/usr/bin/env python3
"""Make a read-only video browsing derivative of the existing image-based sim export."""
import argparse
import hashlib
import io
import json
import shutil
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = args.run / "data/final/lerobot-0.4.4"
    info_path = source / "meta/info.json"
    info = json.loads(info_path.read_text())
    fps = int(info["fps"])
    cameras = [k for k, v in info["features"].items() if v["dtype"] == "image"]
    assert len(cameras) == 3 and fps in (20, 30)
    assert not args.out.exists(), "Use a new output directory; no replay over existing output"
    av.CodecContext.create("libx264", "w")
    source_files = sorted(p for folder in ("meta", "data") for p in (source / folder).rglob("*") if p.is_file())
    source_stat = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in source_files}
    protected = [info_path, source / "export-provenance.json",
                 args.run / "data/final/prepared/manifest.json"]
    binary_root = args.run / "data/final/binary-view"
    binary_acceptance = binary_root / "acceptance.json"
    binary_split = binary_root / "sim-binary-task-view.npz"
    assert binary_acceptance.exists() == binary_split.exists(), "Incomplete binary view"
    if binary_acceptance.exists():
        protected.extend([binary_acceptance, binary_split])
    source_hash = {str(p): sha(p) for p in protected}
    args.out.mkdir(parents=True)
    progress = {"status": "encoding", "source_root": str(source), "output_root": str(args.out),
                "new_simulation_steps": 0, "new_training_updates": 0, "new_llm_requests": 0,
                "videos": [], "source_sha256_before": source_hash}
    save(args.out / "preview-progress.json", progress)

    for p in (source / "meta").rglob("*"):
        if p.is_file() and p.suffix != ".parquet" or p.is_file() and "episodes" not in p.parts:
            if p == info_path:
                continue
            target = args.out / p.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)

    numeric_files = []
    for p in sorted((source / "data").rglob("*.parquet")):
        cols = [k for k in pq.ParquetFile(p).schema_arrow.names if k not in cameras]
        table = pq.read_table(p, columns=cols)
        target = args.out / p.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, target, compression="snappy")
        assert pq.read_table(target).equals(table, check_metadata=False)
        numeric_files.append({"path": str(p.relative_to(source)), "rows": len(table),
                              "all_non_image_columns_exactly_equal": True})

    episode_rows = []
    for p in sorted((source / "meta/episodes").rglob("*.parquet")):
        table = pq.read_table(p)
        rows = table.to_pylist()
        episode_rows.extend(rows)
        for cam in cameras:
            for suffix, values in [
                ("chunk_index", [0] * len(rows)),
                ("file_index", [int(row["episode_index"]) for row in rows]),
                ("from_timestamp", [0.0] * len(rows)),
                ("to_timestamp", [int(row["length"]) / fps for row in rows]),
            ]:
                table = table.append_column(f"videos/{cam}/{suffix}", pa.array(values))
        target = args.out / p.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, target)

    preview_info = json.loads(json.dumps(info))
    preview_info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    preview_info["total_videos"] = len(episode_rows) * len(cameras)
    for cam in cameras:
        preview_info["features"][cam]["dtype"] = "video"
        preview_info["features"][cam]["video_info"] = {
            "video.fps": fps, "video.codec": "h264", "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False, "video.height": 240, "video.width": 320,
        }
    preview_info["viewing_derivative"] = {
        "purpose": "Read-only browsing; training continues to use the original image-based export.",
        "images": "Existing exported Parquet image bytes decoded and encoded to H.264 CRF18; no rendering or physics.",
        "numeric_data": "Every non-image column is retained with exact Arrow value equality.",
        "source": str(source), "new_simulation_steps": 0,
    }
    save(args.out / "meta/info.json", preview_info)

    exports = json.loads((source / "export-provenance.json").read_text())["episodes"]
    split_by_episode = {}
    if binary_split.exists():
        split_npz = np.load(binary_split, allow_pickle=False)
        for index in np.unique(split_npz["episode_index"]):
            labels = np.unique(split_npz["split"][split_npz["episode_index"] == index])
            assert len(labels) == 1
            split_by_episode[int(index)] = str(labels[0])
    else:
        for episode in exports:
            label = episode["scenario_record"]["scenario"]["split"]
            assert isinstance(label, str) and label, "Source scenario split is required"
            split_by_episode[int(episode["episode_index"])] = label
    catalog = []
    for row in episode_rows:
        index = int(row["episode_index"])
        origin = next(e for e in exports if int(e["episode_index"]) == index)
        catalog.append({"episode_index": index, "frames": int(row["length"]),
                        "duration_s": int(row["length"]) / fps,
                        "split": split_by_episode[index], "source_episode_id": origin["source_episode_id"],
                        "source_task_id": origin["source_task_id"],
                        "source_scene": origin.get("scenario_record"),
                        "source_data_file": info["data_path"].format(chunk_index=row["data/chunk_index"], file_index=row["data/file_index"])})
    save(args.out / "preview-episode-index.json", catalog)
    rows_ordered = sorted(episode_rows, key=lambda row: (int(row["episode_index"]) != 3, int(row["episode_index"])))
    image_checks = []
    for row in rows_ordered:
        index, length = int(row["episode_index"]), int(row["length"])
        data_path = source / info["data_path"].format(chunk_index=row["data/chunk_index"], file_index=row["data/file_index"])
        containers, streams, paths = {}, {}, {}
        for cam in cameras:
            target = args.out / "videos" / cam / "chunk-000" / f"file-{index:03d}.mp4"
            target.parent.mkdir(parents=True, exist_ok=True)
            container = av.open(str(target), "w", format="mp4", options={"movflags": "+faststart"})
            stream = container.add_stream("libx264", rate=fps, options={"crf": "18", "preset": "ultrafast"})
            stream.width, stream.height, stream.pix_fmt = 320, 240, "yuv420p"
            stream.codec_context.thread_count = 2
            containers[cam], streams[cam], paths[cam] = container, stream, target
        count = 0
        for batch in pq.ParquetFile(data_path).iter_batches(batch_size=32, columns=["episode_index", "frame_index", *cameras]):
            for record in batch.to_pylist():
                if int(record["episode_index"]) != index:
                    continue
                assert int(record["frame_index"]) == count
                for cam in cameras:
                    image = record[cam]
                    im = Image.open(io.BytesIO(image["bytes"])) if image.get("bytes") is not None else Image.open(source / image["path"])
                    im = im.convert("RGB")
                    assert im.size == (320, 240)
                    frame = av.VideoFrame.from_image(im)
                    frame.pts = count
                    frame.time_base = Fraction(1, fps)
                    for packet in streams[cam].encode(frame):
                        containers[cam].mux(packet)
                    if count in (0, length // 2, length - 1):
                        image_checks.append({"episode": index, "camera": cam, "frame": count,
                                             "source_rgb_sha256": hashlib.sha256(im.tobytes()).hexdigest(),
                                             "source_rgb_std": float(np.asarray(im).std())})
                count += 1
        assert count == length
        for cam in cameras:
            for packet in streams[cam].encode():
                containers[cam].mux(packet)
            containers[cam].close()
            with av.open(str(paths[cam])) as check:
                stream = check.streams.video[0]
                assert stream.frames == length and stream.average_rate == fps
                duration = float(check.duration / av.time_base)
                assert abs(duration - length / fps) < 0.051
            progress["videos"].append({"episode_index": index, "camera": cam,
                                        "path": str(paths[cam].relative_to(args.out)), "frames": length,
                                        "duration_s": duration, "bytes": paths[cam].stat().st_size})
        save(args.out / "preview-progress.json", progress)
        print(json.dumps({"episode_complete": index, "frames": length, "split": split_by_episode[index],
                          "videos_complete": len(progress["videos"]), "total_videos": len(episode_rows) * 3}), flush=True)

    assert len(progress["videos"]) == len(episode_rows) * 3
    assert sum(v["frames"] for v in progress["videos"]) == int(info["total_frames"]) * 3
    source_hash_after = {str(p): sha(p) for p in protected}
    source_stat_after = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in source_files}
    assert source_hash_after == source_hash and source_stat_after == source_stat
    progress.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                    source_sha256_after=source_hash_after, source_stat_unchanged=True,
                    numeric_files=numeric_files, image_samples=image_checks,
                    source_episodes=len(episode_rows), source_frames=int(info["total_frames"]), fps=fps,
                    camera_count=3, original_training_data_modified=False,
                    video_encoding={"codec": "libx264", "crf": 18, "preset": "ultrafast", "pixel_format": "yuv420p"})
    save(args.out / "preview-acceptance.json", progress)
    save(args.out / "preview-progress.json", progress)
    print(json.dumps({"status": "completed", "out": str(args.out), "videos": len(progress["videos"])}), flush=True)


if __name__ == "__main__":
    main()
