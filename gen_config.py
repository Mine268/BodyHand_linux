# 仅使用系统库解析命令行参数
import sys
import os

def parse_args():
    args = sys.argv[1:]  # 排除脚本名称
    config = {}

    for arg in args:
        if '=' in arg:
            key, value = arg.split('=', 1)
            config[key] = value
        else:
            config[arg] = True  # 处理无值参数，默认为True

    return config

if __name__ == "__main__":
    config = parse_args()

    if "--help" in config or "-h" in config:
        print('''python gen_config.py <stereo_calib_file> <mono_calib_file>''')
        exit(0)

    # 传入两个参数：双视图标定文件和单视图标定文件
    if len(sys.argv) != 3:
        print("Error: Please provide exactly two arguments: <stereo_calib_file> <mono_calib_file>")
        exit(1)

    stereo_calib_file = sys.argv[1]
    mono_calib_file = sys.argv[2]

    # 确保文件存在而且是文件类型
    if not os.path.isfile(stereo_calib_file):
        print(f"Error: Stereo calibration file '{stereo_calib_file}' does not exist or is not a file.")
        exit(1)
    if not os.path.isfile(mono_calib_file):
        print(f"Error: Mono calibration file '{mono_calib_file}' does not exist or is not a file.")
        exit(1)

    # 直接读取文件的全部内容
    stereo_calib_data = ""
    mono_calib_data = ""
    with open(stereo_calib_file, 'r') as f:
        stereo_calib_data = f.read()
    with open(mono_calib_file, 'r') as f:
        mono_calib_data = f.read()

    # 预处理一下双视图的标定数据
    # 将25-27行的数据读出来，这是标定好的3*4 RT矩阵 [a11,a12,a13,tx; a21,a22,a23,ty; a31,a32,a33,tz]
    # 将他读取为一维list
    stereo_lines = stereo_calib_data.splitlines()
    rt_matrix = []
    for i in range(24, 27):
        line = stereo_lines[i].strip()
        values = list(map(float, stereo_lines[i][1:-1].split(", ")))
        rt_matrix.extend(values)
    r_matrix = rt_matrix[0:3] + rt_matrix[4:7] + rt_matrix[8:11]
    t_matrix = rt_matrix[3:4] + rt_matrix[7:8] + rt_matrix[11:12]

    # 输出到calib.cfg配置文件
    with open("calib.cfg", "w") as f:
        f.write("[Human detection model file]\n")
        f.write("[Pose model file]\n")
        f.write("[Hand detection model file]\n")
        f.write("[Hand pose estimation model file]\n")
        f.write("2\n")  # 默认双目摄像头

        # 开始写入标定数据
        # 26个浮点数，前9个表示内参矩阵，接着的9个表示旋转变换矩阵，然后3个表示位移变换向量，最后的5个表示畸变参数
        # 目前用的是写死的内参，第一个视图默认用的是原点位姿，姿态估计程序自己会根据单视图标定的结果自己算
        f.write("1052.721 0 730.726 0 1047.504 548.518 0 0 1 ")
        f.write("1 0 0 0 1 0 0 0 1 0 0 0 ")
        f.write("0 0 0 0 0\n")
        # 写入第二个相机的标定数据
        f.write("1052.721 0 730.726 0 1047.504 548.518 0 0 1 ")
        f.write(" ".join(map(str, r_matrix)) + " " + " ".join(map(str, t_matrix)) + " ")
        f.write("0 0 0 0 0\n")
        # 写入单视图标定数据
        f.write(mono_calib_data + "\n")

    print("Calibration configuration 'calib.cfg' generated successfully.")
    print("Please edit the model file paths in 'calib.cfg' before using it.")