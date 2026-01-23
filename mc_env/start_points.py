from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class StartPoint:
    id: str
    weight: float
    x: int
    y: int
    z: int
    yaw: float
    pitch: float
    goalX: int
    goalY: int
    goalZ: int

    def to_json(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "weight": self.weight,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "yaw": self.yaw,
            "pitch": self.pitch,
            "goalX": self.goalX,
            "goalY": self.goalY,
            "goalZ": self.goalZ,
        }
