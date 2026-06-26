import streamlit as st
import folium
from streamlit_folium import st_folium
from pyvis.network import Network
import pandas as pd
import numpy as np
import time
from datetime import datetime

# ===================== 坐标系转换工具 GCJ-02 <=> WGS84 =====================
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

# ===================== 心跳包模拟类 =====================
class HeartBeat:
    def __init__(self):
        self.seq_num = 0
        self.last_time = time.time()
        self.data_list = []
        self.timeout_limit = 3

    def create_heartbeat(self):
        self.seq_num += 1
        now_stamp = time.time()
        now_time_text = datetime.now().strftime("%H:%M:%S")
        pkg = {"序号": self.seq_num, "时刻": now_time_text, "时间戳": now_stamp}
        self.data_list.append(pkg)
        self.last_time = now_stamp
        return pkg

    def check_offline(self):
        return time.time() - self.last_time > self.timeout_limit

# ===================== 页面基础配置 =====================
st.set_page_config(page_title="无人机智能监控系统", layout="wide")

# 侧边菜单栏
with st.sidebar:
    st.title("功能导航")
    select_page = st.radio("页面选择", ["航线规划", "飞行监控", "通信拓扑"])
    st.divider()
    st.subheader("坐标系设置")
    coord_system = st.radio("输入坐标系", ["WGS-84", "GCJ-02(高德/百度)"])
    st.divider()
    st.subheader("系统运行状态")
    st.success("起点A已配置")
    st.success("终点B已配置")

# ===================== 页面1：航线规划（地图+坐标设置） =====================
if select_page == "航线规划":
    st.header("航线规划页面｜校园地图航点设置")
    map_col, ctrl_col = st.columns([3, 1])

    with ctrl_col:
        st.subheader("点位控制面板")
        st.markdown("**起点A 经纬度**")
        latA = st.number_input("纬度A", value=32.2322, step=0.0001)
        lonA = st.number_input("经度A", value=118.7490, step=0.0001)
        st.checkbox("启用A点标记", value=True)

        st.markdown("**终点B 经纬度**")
        latB = st.number_input("纬度B", value=32.2343, step=0.0001)
        lonB = st.number_input("经度B", value=118.7440, step=0.0001)
        st.checkbox("启用B点标记", value=True)

        fly_height = st.slider("飞行高度(m)", min_value=10, max_value=100, value=50)
        st.info(f"当前设定飞行高度：{fly_height} 米")

    with map_col:
        # 坐标统一转换成WGS84给到地图
        if coord_system == "GCJ-02(高德/百度)":
            wgs_a_lat, wgs_a_lon = gcj02_to_wgs84(latA, lonA)
            wgs_b_lat, wgs_b_lon = gcj02_to_wgs84(latB, lonB)
        else:
            wgs_a_lat, wgs_a_lon = latA, lonA
            wgs_b_lat, wgs_b_lon = latB, lonB

        # 初始化卫星地图
        map_obj = folium.Map(location=[wgs_a_lat, wgs_a_lon], zoom_start=16, tiles="OpenStreetMap")
        folium.Marker([wgs_a_lat, wgs_a_lon], icon=folium.Icon(color="red"), popup="起点A").add_to(map_obj)
        folium.Marker([wgs_b_lat, wgs_b_lon], icon=folium.Icon(color="green"), popup="终点B").add_to(map_obj)
        st_folium(map_obj, height=620, width="100%")

# ===================== 页面2：飞行监控 心跳包监测 =====================
elif select_page == "飞行监控":
    st.header("飞行监控页面｜无人机心跳包实时检测")
    beat_ins = HeartBeat()
    chart_container = st.empty()
    warn_box = st.empty()
    table_container = st.empty()

    # 持续刷新心跳数据
    while True:
        packet = beat_ins.create_heartbeat()
        df_data = pd.DataFrame(beat_ins.data_list)

        # 断线判定提示
        if beat_ins.check_offline():
            warn_box.error("⚠️ 警告：超过3秒未接收心跳，无人机失联！")
        else:
            warn_box.success(f"✅ 设备在线，最新心跳包序号：{packet['序号']}")

        # 绘制时序折线图
        with chart_container:
            st.line_chart(df_data, x="序号", y="时间戳")
        # 展示最新表格数据
        with table_container:
            st.dataframe(df_data.tail(12), use_container_width=True)

        time.sleep(1)

# ===================== 页面3：通信拓扑 GCS-OBC-FCU =====================
elif select_page == "通信拓扑":
    st.header("三层通信链路拓扑：地面站GCS - 机载OBC - 飞控FCU")
    # 绘制拓扑网络图
    network = Network(height="340", width="100%")
    network.add_node("GCS", label="GCS地面站\n192.168.1.100", shape="box", color="#4285F4")
    network.add_node("OBC", label="OBC树莓派机载计算机", shape="ellipse", color="#FFCC44")
    network.add_node("FCU", label="PX4机载飞控FCU", shape="square", color="#BB77DD")
    network.add_edge("GCS", "OBC", label="UDP 14550")
    network.add_edge("OBC", "FCU", label="MAVLink协议")
    st.components.v1.html(network.generate_html(), height=350)

    st.info("链路整体状态：通信正常，往返延迟≈25ms，丢包率0.1%")

    # 双向数据流标签页
    tab1, tab2 = st.tabs(["下行：GCS下发航线任务", "上行：飞控回传飞行状态"])
    down_log = pd.DataFrame([
        {"时间":"14:36:01","日志":"完成A*航线规划，航点数量10，检测障碍物6处"},
        {"时间":"14:36:07","日志":"更新航线参数，调整飞行路径总长"}
    ])
    up_log = pd.DataFrame([
        {"时间":"15:03:08","日志":"飞机切入自动飞行模式"},
        {"时间":"15:03:10","日志":"抵达航点1，依次遍历全部9个航点"},
        {"时间":"15:03:51","日志":"全部航点飞行完毕，任务执行完成"}
    ])
    with tab1:
        st.dataframe(down_log, use_container_width=True)
    with tab2:
        st.dataframe(up_log, use_container_width=True)
