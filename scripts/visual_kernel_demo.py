from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from living_tabletop.visual_kernel.service import VisualWorldService


def main() -> int:
    with TemporaryDirectory(prefix="living-tabletop-vwk-") as directory:
        service = VisualWorldService(db_path=Path(directory) / "world.db")
        created = service.create_session()
        session_id = created["session_id"]
        version = created["projection"]["world_version"]
        steps = [
            ("move", {"destination_id": "loc_foyer"}, "进入门厅"),
            ("move", {"destination_id": "loc_hall"}, "进入走廊"),
            ("inspect", {"target_id": "obj_cellar_marks"}, "调查拖痕"),
            ("interact", {"target_id": "obj_brass_key", "verb": "take"}, "拿起钥匙"),
            ("interact", {"target_id": "conn_hall_basement", "verb": "unlock"}, "首次解锁"),
            ("interact", {"target_id": "conn_hall_basement", "verb": "unlock"}, "再次解锁"),
            ("interact", {"target_id": "conn_hall_basement", "verb": "open"}, "打开暗门"),
            ("move", {"destination_id": "loc_basement"}, "进入地下室"),
        ]
        print(f"session={session_id}")
        for index, (kind, payload, label) in enumerate(steps, start=1):
            result = service.command(
                session_id,
                kind=kind,
                payload=payload,
                expected_state_version=version,
                idempotency_key=f"demo-step-{index}",
            )
            receipt = result["receipt"]
            projection = result["projection"]
            version = projection["world_version"]
            print(
                f"{index:02d} {label}: {receipt['outcome']} "
                f"version={version} time={projection['world']['time']}"
            )
        report = service.replay(session_id)
        final_projection = service.projection(session_id)
        assert final_projection["observer"]["location_id"] == "loc_basement"
        assert report["verified"] is True
        print(
            f"PASS location=loc_basement events={report['event_count']} "
            f"replay_verified={report['verified']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
