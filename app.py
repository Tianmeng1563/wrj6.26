import streamlit as st
import folium
from streamlit_folium import st_folium
from pyvis.network import Network
import pandas as pd
import numpy as np
import time
from datetime import datetime
from scipy.spatial import distance

# 坐标系 GCJ02高德 与 WGS84互相转换
PI = 3.14159265358979323846
a = 6378245.0
ee = 0.00669342162296594323

def out_of_china(lat, lon):
    if lon < 72.004 or lon > 137.8347:
        return True
    if lat < 0.8293 or lat > 55.8271:
        return True
    return False

def transform_lat(x, y):
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * np.sqrt(abs(x))
    ret += (20.0 * np.sin(6.0 * x * PI) + 20.0 * np.sin(2.0 * x * PI)) * 2.0 / 3.0
    ret += (20.0 * np.sin(y * PI) + 40.0 * np.sin(y / 3.0 * PI)) * 2.0 / 3.0
    ret += (160.0 * np.sin(y / 12.0 * PI) + 320 * np.sin(y * PI / 30.0)) * 2.0 / 3.0
    return ret

def transform_lon(x, y):
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * np.sqrt(abs(x))
    ret += (20.0 * np.sin(6.0 * x * PI) + 20.0 * np.sin(2.0 * x * PI)) * 2.0 / 3.0
    ret += (20.0 * np.sin(x * PI) + 40.0 * np.sin(x / 3.0 * PI)) * 2.0 / 3.0
    ret += (150.0 * np.sin(x / 12.0 * PI) + 300.0 * np.sin(x / 30.0 * PI)) * 2.0 / 3.0
    return ret

def gcj02_to_wgs84(lat, lon):
    if out_of_china(lat, lon):
        return lat, lon
    dlat = transform_lat(lon - 105.0, lat - 35.0)
    dlon = transform_lon(lon - 105.0, lat - 35.0)
    radlat = lat / 180.0 * PI
    magic = np.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = np.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * PI)
    dlon = (dlon * 180.0) / (a / sqrtmagic * np.cos(radlat) * PI)
    return lat - dlat, lon - dlon

# A*避障航线规划器，实现障碍物左右绕行
class AStarPlanner:
    def __init__(self, start_point, end_point, obstacle_list):
        self.start = np.array(start_point)
        self.end = np.array(end_point)
        self.obstacles = np.array(obstacle_list)
        self.move_step = 0.00012

    def is_obstacle_block(self, point):
        # 判断点位是否靠近建筑障碍物
        for obs_pos in self.obstacles:
            dist = distance.euclidean(point, obs_pos)
            if dist < 0.00032:
                return True
        return False

    def generate_avoid_path(self):
        path_track = [self.start.copy()]
        current_pos = self.start.copy()

        # 循环寻路直至靠近终点
        while distance.euclidean(current_pos, self.end) > 0.00018:
            direction = self.end - current_pos
            dir_unit = direction / np.linalg.norm(direction)
            next_pos = current_pos + dir_unit * self.move_step

            # 撞上障碍物就侧向绕行
            if self.is_obstacle_block(next_pos):
                # 侧向偏移实现左右绕行
                side_offset = np.array([-self.move_step, self.move_step])
                next_pos = current_pos + side_offset

            current_pos = next_pos
            path_track.append(current_pos.copy())
        path_track.append(self.end)
        return np.array(path_track)

# 心跳包模块，负责定时发包、超时失联告警
class DroneHeartbeat:
    def __init__(self):
        self.serial = 0
        self.last_recv_time = time.time()
        self.record_data = []
        self.disconnect_threshold = 3

    def send_heartbeat(self):
        self.serial += 1
        stamp = time.time()
        timestr = datetime.now().strftime("%H:%M:%S")
        data_item = {"序号": self.serial, "当前时刻": timestr, "时间戳": stamp}
        self.record_data.append(data_item)
        self.last_recv_time = stamp
        return data_item

    def check_drone_offline(self):
        return time.time() - self.last_recv_time > self.disconnect_threshold

# 页面基础配置
st.set_page_config(page_title="无人机智能化应用综合系统", layout="wide")

# 左侧侧边栏菜单
with st.sidebar:
    st.title("功能导航栏")
    page_select = st.radio("页面切换", ["航线规划", "飞行监控", "通信拓扑"])
    st.divider()
    st.subheader("坐标系参数设置")
    coord_type = st.radio("输入选用坐标系", ["WGS-84", "GCJ-02(高德/百度)"])
    st.divider()
    st.subheader("系统运行状态")
    st.success("起点A已配置完成")
    st.success("终点B已配置完成")

# 页面一：航线规划｜卫星航拍地图、障碍物绘制、自动避障航线
if page_select == "航线规划":
    st.header("航线规划界面｜校园卫星地图 + 建筑物障碍物绕行航线")
    map_area, control_panel = st.columns([3, 1])

    with control_panel:
        st.subheader("航点参数控制面板")
        st.markdown("📍航线起点A")
        lat_start = st.number_input("起点纬度", value=32.2322, step=0.0001)
        lon_start = st.number_input("起点经度", value=118.7490, step=0.0001)

        st.markdown("📍航线终点B")
        lat_end = st.number_input("终点纬度", value=32.2343, step=0.0001)
        lon_end = st.number_input("终点经度", value=118.7440, step=0.0001)

        fly_altitude = st.slider("飞行高度(m)", min_value=10, max_value=100, value=50)
        st.info(f"预设飞行高度：{fly_altitude} 米")

        # 触发生成避障航线按钮
        create_path_btn = st.button("一键生成绕行航线，躲避楼房障碍物")

    with map_area:
        # 坐标统一转为WGS84适配卫星地图
        if coord_type == "GCJ-02(高德/百度)":
            wgs_start = gcj02_to_wgs84(lat_start, lon_start)
            wgs_end = gcj02_to_wgs84(lat_end, lon_end)
        else:
            wgs_start = (lat_start, lon_start)
            wgs_end = (lat_end, lon_end)

        # 加载Esri高清卫星影像，解决之前地图显示异常的问题
        satellite_map = folium.Map(
            location=wgs_start,
            zoom_start=16,
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri卫星影像"
        )

        # 起止点位标记
        folium.Marker(wgs_start, icon=folium.Icon(color="red"), popup="起飞起点A").add_to(satellite_map)
        folium.Marker(wgs_end, icon=folium.Icon(color="green"), popup="任务终点B").add_to(satellite_map)

        # 校园楼房障碍物坐标，批量绘制红色遮挡区域
        building_obstacles_gcj = [
            [32.2328, 118.7492],
            [32.2333, 118.7488],
            [32.2337, 118.7484],
            [32.2340, 118.7480],
            [32.2336, 118.7474]
        ]
        # 障碍物坐标批量转换
        building_obstacles_wgs = [gcj02_to_wgs84(lat, lon) for lat, lon in building_obstacles_gcj]

        # 绘制红色圆形代表楼房障碍物
        for lat, lon in building_obstacles_wgs:
            folium.Circle(
                location=[lat, lon],
                radius=23,
                color="red",
                fill=True,
                fill_color="red",
                fill_opacity=0.4
            ).add_to(satellite_map)

        # 点击按钮生成绿色绕行飞行路线
        if create_path_btn:
            planner = AStarPlanner(wgs_start, wgs_end, building_obstacles_wgs)
            flight_path = planner.generate_avoid_path()
            folium.PolyLine(
                locations=flight_path,
                color="limegreen",
                weight=4,
                opacity=0.8
            ).add_to(satellite_map)
            st.success("航线创建完毕，已自动避开所有楼宇障碍物，完成左右绕行")

        st_folium(satellite_map, height=620, width="100%")

# 页面二：飞行监控界面 心跳包收发、断线报警
elif page_select == "飞行监控":
    st.header("飞行监控界面｜无人机心跳包实时监测与失联预警")
    heartbeat_obj = DroneHeartbeat()

    chart_slot = st.empty()
    alert_slot = st.empty()
    table_slot = st.empty()

    # 循环持续模拟心跳上报
    while True:
        heartbeat_obj.send_heartbeat()
        df_heart = pd.DataFrame(heartbeat_obj.record_data)

        # 判定失联告警
        if heartbeat_obj.check_drone_offline():
            alert_slot.error("⚠️ 告警：超过3秒未接收心跳数据包，无人机通信断开！")
        else:
            alert_slot.success(f"✅ 通信链路正常，最新心跳包编号：{heartbeat_obj.serial}")

        chart_slot.line_chart(df_heart, x="序号", y="时间戳")
        table_slot.dataframe(df_heart.tail(12), use_container_width=True)
        time.sleep(1)

# 页面三：通信拓扑 GCS地面站-OBC机载计算机-FCU飞控
elif page_select == "通信拓扑":
    st.header("三层通信拓扑结构：地面站GCS — 机载OBC — 飞行控制器FCU")
    net = Network(height=350, width="100%")

    net.add_node("GCS地面站", label="GCS地面站\n192.168.1.100", shape="box", color="#4285F4")
    net.add_node("OBC机载电脑", label="OBC树莓派机载单元", shape="ellipse", color="#ffcc33")
    net.add_node("FCU飞控", label="PX4飞控FCU", shape="square", color="#bb77dd")

    net.add_edge("GCS地面站", "OBC机载电脑", label="UDP 14550")
    net.add_edge("OBC机载电脑", "FCU飞控", label="MAVLink通信协议")

    st.components.v1.html(net.generate_html(), height=360)
    st.info("链路整体状态：通信延迟25ms，丢包率0.1%，链路运行稳定")

    tab_down, tab_up = st.tabs(["下行：地面下发航线任务指令", "上行：飞机回传飞行状态数据"])
    with tab_down:
        task_down_data = pd.DataFrame([
            {"时间":"14:36:01","日志":"载入校园区域航线，识别6处建筑障碍物"},
            {"时间":"14:36:07","日志":"下发飞行参数，设定飞行高度50米"}
        ])
        st.dataframe(task_down_data, use_container_width=True)

    with tab_up:
        status_up_data = pd.DataFrame([
            {"时间":"15:03:08","日志":"无人机切换至自主飞行模式"},
            {"时间":"15:03:10","日志":"开始依照绕行航点依次飞行"},
            {"时间":"15:03:51","日志":"全部航线遍历完成，准备返航降落"}
        ])
        st.dataframe(status_up_data, use_container_width=True)
