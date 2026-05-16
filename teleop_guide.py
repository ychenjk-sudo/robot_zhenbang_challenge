"""
Teleoperation Calibration Helper
Use keyboard or simple prompts to find good joint angles for each pose.
No robot connection needed for planning; this just helps you record numbers.
"""
import yaml
from pathlib import Path


def teleop_guide():
    print("=" * 60)
    print("LeRobot 真棒挑战 - 姿态标定助手")
    print("=" * 60)
    print("""
建议步骤 (使用 LeRobot 官方的遥操作来手动控制机械臂):

1. 先运行官方遥操作:  
   python lerobot/scripts/control_robot.py \
       --robot.type=so100 \
       --control.type=teleoperate \
       --control.fps=30

2. 用手持主臂(leader)控制从臂(follower)到各个目标位置

3. 记录下从臂此时的关节角度 (从终端输出或 observation 中读取)

4. 将数值填入 config.yaml 的 game.poses 下对应姿态

推荐姿态:
- home:       手臂抬高俯视桌面，摄像头能清楚看到两个物品
- left_hover:  机械臂末端悬停在左侧物品正上方约3-5cm
- left_hit:    手腕稍微下压，做出'拍'的动作 (wrist_flex 增大)
- right_hover: 同上，右侧
- right_hit:   同上，右侧拍击
- left_peek:   假装要选左侧但没完全过去 (用于fake-out)
- right_peek:  同上
- center_look: 左右犹豫时的中心位置
- celebrate:   抬高+夹爪开合的庆祝动作
- sad:         耷拉下来的沮丧动作

提示:
- shoulder_pan:  控制左右旋转 (负数左，正数右)
- shoulder_lift:  控制手臂抬起/放下
- elbow_flex:     控制肘部弯曲
- wrist_flex:     控制手腕俯仰 (拍击主要靠这个!)
- wrist_roll:     旋转末端
- gripper:        夹爪开合 (0=开, 100=合)
""")

    # Try to load current config and print poses for reference
    cfg_path = Path("config.yaml")
    if cfg_path.exists():
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        poses = cfg.get("game", {}).get("poses", {})
        print("\n当前 config.yaml 中的姿态参考值:")
        for name, vals in poses.items():
            print(f"\n  {name}:")
            for k, v in vals.items():
                print(f"    {k}: {v}")
    else:
        print("\nconfig.yaml 不在当前目录，请确认路径。")

    print("\n" + "=" * 60)
    input("按 Enter 退出...")


if __name__ == "__main__":
    teleop_guide()
