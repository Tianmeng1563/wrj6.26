import streamlit as st
import folium
from streamlit_folium import st_folium
from streamlit_option_menu import option_menu
from datetime import datetime, timedelta
import time
import pandas as pd
import json
import os
import numpy as np
from shapely.geometry import LineString, Polygon

# 页面配置必须在最前面
st.set_page_config(layout="wide", page_title="南科院无人机航线规划")

SAVE_FILE = "drone_data.json"

def load_all_data():
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE,"r",encoding="utf-8") as f:
            return json.load(f)
    # ===================== 南科院校内精准坐标 都在校园里面 =====================
    return {
        "A": [32.2346, 118.7492],   # 校内操场
        "B": [32.2335, 118.7505],   # 校内实训楼
        "A_set": True,
        "B_set": True,
        "obstacles": []
    }

def save_all_data():
    data={
        "A":list(st.session_state.A),"B":list(st.session_state.B),
        "A_set":st.session_state.A_set,"B_set":st.session_state.B_set,
        "obstacles":st.session_state.polygon_memory
    }
    with open(SAVE_FILE,"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)

data=load_all_data()
default_states = {
    "A": tuple(data["A"]), "B": tuple(data["B"]),
    "A_set": data["A_set"], "B_set": data["B_set"],
    "height": 50, "heartbeat_data": [], "polygon_memory": data["obstacles"],
    "is_drawing": False, "temp_points": [], "obs_h": 20, "last_click_time": 0,
    "safe_radius": 0.0002,
    "flight_running": False, "flight_paused": False, "current_wp_idx": 0,
    "flight_speed": 8.5, "flight_start_time": None, "flight_waypoints": [],
    "battery": 100.0, "total_distance": 0.0, "elapsed_distance": 0.0,
    "route_side": "auto"
}

for key, val in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --------------------------通信拓扑模块--------------------------
st.header("通信链路拓扑与数据流")
# 设备在线状态
col_gcs, col_obc, col_fcu = st.columns(3)
with col_gcs:
    st.checkbox("GCS在线", value=True, disabled=True)
with col_obc:
    st.checkbox("OBC在线", value=True, disabled=True)
with col_fcu:
    st.checkbox("FCU在线", value=True, disabled=True)

# 链路连接信息文本展示
st.subheader("链路连接信息")
link_col1, link_col2, link_col3 = st.columns(3)
with link_col1:
    st.markdown("**GCS 地面站**")
    st.text("IP：192.168.1.100")
with link_col2:
    st.markdown("**OBC 机载计算机**")
    st.text("通信协议：UDP 14550")
    st.text("设备：Raspberry Pi 4")
with link_col3:
    st.markdown("**FCU 飞控单元**")
    st.text("通信协议：MAVLink")
    st.text("固件：PX4 / ArduPilot")

# 链路统计信息
st.info("链路统计：GCS↔OBC：通信正常 | OBC↔FCU：通信正常 | 链路延迟：~25ms | 丢包率：0.1%")

# 通信日志标签页
st.subheader("通信日志")
tab_business, tab_gof, tab_fog = st.tabs(["业务流程", "GCS→OBC→FCU", "FCU→OBC→GCS"])
with tab_business:
    st.text_area("业务流程日志", """航线规划完成 | type:horizontal | 航点数:9 | 路径长度:359.8m
OBC 内部
[14:36:01.607] 航线规划执行成功
航线规划完成 | type:horizontal | 航点数:10 | 路径长度:356.3m
OBC 内部
[14:32:54.650] 开始航线规划 | 算法:A* | 障碍物数量:6
导航目标：起点(32.234368, 118.744358) 终点(32.236468, 118.744058) 高度10m
GCS下发航线指令至OBC""", height=220)
with tab_gof:
    st.text_area("下行通信日志", """[15:03:03.08] GCS→OBC→FCU：任务启动指令
[15:03:10] OBC转发导航指令至FCU
[15:03:17] FCU接收航点1执行指令
[15:03:20] FCU接收航点2执行指令""", height=220)
with tab_fog:
    st.text_area("上行通信日志", """[15:03:03.08] FCU→OBC→GCS：ACK 飞行模式AUTO
[15:03:10] FCU→OBC→GCS：WP_REACHED #1
[15:03:17] FCU→OBC→GCS：WP_REACHED #2
[15:03:51] FCU→OBC→GCS：MISSION_COMPLETE
OBC汇总状态上传至GCS""", height=220)
st.divider()
# ----------------------------------------------------------------------------------------

# 航线偏移、避障计算
def calc_route_lines(pA,pB,offset=0.0001):
    latA,lonA=pA
    latB,lonB=pB
    dx=lonB-lonA
    dy=latB-latA
    L=np.hypot(dx,dy)
    if L<1e-8:L=1e-8
    left_off_x=-dy/L*offset
    left_off_y=dx/L*offset
    right_off_x=dy/L*offset
    right_off_y=-dx/L*offset
    left=[[latA,lonA],[latA+left_off_y,lonA+left_off_x],[latB+left_off_y,lonB+left_off_x],[latB,lonB]]
    right=[[latA,lonA],[latA+right_off_y,lonA+right_off_x],[latB+right_off_y,lonB+right_off_x],[latB,lonB]]
    return left,right

def get_safe_route(pA, pB, obstacles, safe_dist, route_side="auto"):
    base_line = LineString([pA, pB])
    obs_polygons = []
    for obs in obstacles:
        pts = obs["pts"]
        if len(pts)>=3:
            poly = Polygon(pts).buffer(safe_dist)
            obs_polygons.append(poly)
    conflict = False
    for poly in obs_polygons:
        if base_line.intersects(poly):
            conflict = True
            break
    if not conflict:
        return [pA, pB], False
    left_line, right_line = calc_route_lines(pA, pB, offset=safe_dist)
    if route_side == "auto":
        left_ok = True
        for poly in obs_polygons:
            if LineString(left_line).intersects(poly):
                left_ok = False
                break
        return (left_line if left_ok else right_line), True
    elif route_side == "left":
        return left_line, True
    else:
        return right_line, True

# 侧边栏
with st.sidebar:
    st.title("🚁 无人机系统导航")
    page=option_menu("功能页面",["航线规划","飞行监控"],default_index=0)
    st.divider()
    st.subheader("系统点位状态")
    st.button("✅ A点已设置" if st.session_state.A_set else "❌ A点未设置",type="primary")
    st.button("✅ B点已设置" if st.session_state.B_set else "❌ B点未设置",type="primary")
    st.divider()
    st.subheader("🛡️ 安全半径配置")
    st.session_state.safe_radius = st.slider("航线与障碍物安全距离", 0.00005, 0.0005, value=st.session_state.safe_radius, step=0.00001, format="%.5f")
    st.session_state.route_side = st.radio("绕飞方向", ["left", "right", "auto"], index=2)

# 航线规划页面
if page=="航线规划":
    st.title("🚁 南京科技职业学院 无人机避障航线规划")
    col_map,col_ctrl=st.columns([3.2,1])
    with col_ctrl:
        st.subheader("🎛️ 点位与飞行参数")
        a_lat=st.number_input("起点A 纬度",value=st.session_state.A[0],format="%.6f")
        a_lon=st.number_input("起点A 经度",value=st.session_state.A[1],format="%.6f")
        b_lat=st.number_input("终点B 纬度",value=st.session_state.B[0],format="%.6f")
        b_lon=st.number_input("终点B 经度",value=st.session_state.B[1],format="%.6f")
        st.session_state.height=st.slider("飞行高度(m)",0,200,value=st.session_state.height)
        if st.button("确定设置起点A"):
            st.session_state.A=(a_lat,a_lon)
            st.session_state.A_set=True
            save_all_data()
            st.success("A点已保存")
        if st.button("确定设置终点B"):
            st.session_state.B=(b_lat,b_lon)
            st.session_state.B_set=True
            save_all_data()
            st.success("B点已保存")
        st.divider()
        st.subheader("🚧 障碍物圈选")
        st.session_state.obs_h=st.number_input("障碍物高度(m)",0,300,value=st.session_state.obs_h)
        if st.session_state.is_drawing:
            st.warning(f"正在绘制，已选点位：{len(st.session_state.temp_points)}")
        else:
            st.info("点击开始绘制，在地图圈禁飞区")
        btn1,btn2,btn3=st.columns(3)
        with btn1:
            if st.button("开始绘制"):
                st.session_state.is_drawing=True
                st.session_state.temp_points=[]
        with btn2:
            if st.button("撤销上一点"):
                if st.session_state.temp_points:st.session_state.temp_points.pop()
        with btn3:
            if st.button("取消绘制"):
                st.session_state.is_drawing=False
                st.session_state.temp_points=[]
        if st.button("✅ 完成圈选保存"):
            if len(st.session_state.temp_points)>=3:
                st.session_state.polygon_memory.append({"pts":st.session_state.temp_points.copy(),"h":st.session_state.obs_h})
                save_all_data()
                st.success("障碍物保存成功")
            else:
                st.error("至少3个点位")
            st.session_state.is_drawing=False
            st.session_state.temp_points=[]
            st.rerun()
        if st.button("🗑️ 清空全部障碍物"):
            st.session_state.polygon_memory=[]
            st.session_state.temp_points=[]
            save_all_data()
            st.rerun()
        st.info(f"已保存障碍物：{len(st.session_state.polygon_memory)} 个")

    with col_map:
        center_lat=(st.session_state.A[0]+st.session_state.B[0])/2
        center_lon=(st.session_state.A[1]+st.session_state.B[1])/2
        # 高德卫星地图
        m=folium.Map(
            location=[center_lat,center_lon],
            zoom_start=19,
            tiles="https://webst01.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}",
            attr="高德卫星地图",
            max_zoom=22
        )
        folium.plugins.Fullscreen(position="topright").add_to(m)

        # 直接用原始校内坐标，不做偏移
        A_raw = st.session_state.A
        B_raw = st.session_state.B

        if st.session_state.A_set:
            folium.Marker(A_raw, icon=folium.Icon(color='red', icon='plane', prefix='fa'), popup="起点A").add_to(m)
        if st.session_state.B_set:
            folium.Marker(B_raw, icon=folium.Icon(color='green', icon='plane', prefix='fa'), popup="终点B").add_to(m)

        for idx,obs in enumerate(st.session_state.polygon_memory):
            pts=obs["pts"]
            hh=obs["h"]
            if len(pts)>=3:
                folium.Polygon(locations=pts,color="#dc2626",fill=True,fill_color="#dc2626",fill_opacity=0.45,popup=f"障碍物{idx+1} | 高度{hh}m").add_to(m)
                poly = Polygon(pts).buffer(st.session_state.safe_radius)
                folium.Polygon(locations=list(poly.exterior.coords), color="#ff9900", fill=False, weight=2, dash_array="5 5").add_to(m)

        if len(st.session_state.temp_points)>0:
            folium.PolyLine(st.session_state.temp_points,color="#ff7700",weight=3,dash_array="10 5").add_to(m)

        if st.session_state.A_set and st.session_state.B_set:
            safe_waypoints, need_avoid = get_safe_route(A_raw, B_raw, st.session_state.polygon_memory, st.session_state.safe_radius, st.session_state.route_side)
            st.session_state.flight_waypoints = safe_waypoints
            folium.PolyLine(safe_waypoints,color="#0066ff",weight=5,popup="校内避障航线").add_to(m)

        output=st_folium(m,width=1150,height=720,key="main_map")
        if st.session_state.is_drawing and output and output.get("last_clicked"):
            now = time.time()
            if now - st.session_state.last_click_time > 0.5:
                pt = output["last_clicked"]
                new_pt = [pt["lat"], pt["lng"]]
                if not st.session_state.temp_points or new_pt != st.session_state.temp_points[-1]:
                    st.session_state.temp_points.append(new_pt)
                    st.session_state.last_click_time = now
                    st.rerun()

# 飞行监控页面
else:
    st.title("📡 飞行实时监控 - 任务执行")
    st.success("✅ 无人机链路正常 设备在线")
    col_btn = st.columns(4)
    with col_btn[0]:
        if st.button("🔴 开始任务", type="primary", disabled=st.session_state.flight_running):
            st.session_state.flight_running = True
            st.session_state.flight_paused = False
            st.session_state.flight_start_time = datetime.now()
            st.session_state.current_wp_idx = 0
            st.rerun()
    with col_btn[1]:
        if st.button("⏸️ 暂停", disabled=not st.session_state.flight_running or st.session_state.flight_paused):
            st.session_state.flight_paused = True
            st.rerun()
    with col_btn[2]:
        if st.button("▶️ 继续", disabled=not st.session_state.flight_paused):
            st.session_state.flight_paused = False
            st.rerun()
    with col_btn[3]:
        if st.button("⏹️ 停止重置", type="secondary"):
            st.session_state.flight_running = False
            st.session_state.flight_paused = False
            st.session_state.current_wp_idx = 0
            st.session_state.battery = 100.0
            st.rerun()

    if len(st.session_state.flight_waypoints) < 2:
        st.warning("⚠️ 先在航线规划页面生成校内航线！")
    else:
        total_dist = 0
        for i in range(len(st.session_state.flight_waypoints)-1):
            p1 = st.session_state.flight_waypoints[i]
            p2 = st.session_state.flight_waypoints[i+1]
            dist = np.hypot(p2[0]-p1[0], p2[1]-p1[1])
            total_dist += dist
        st.session_state.total_distance = round(total_dist * 111000, 2)

        if st.session_state.flight_running and not st.session_state.flight_paused:
            if st.session_state.current_wp_idx < len(st.session_state.flight_waypoints)-1:
                st.session_state.current_wp_idx += 0.01
                st.session_state.battery = max(0, st.session_state.battery - 0.01)
            else:
                st.session_state.flight_running = False
                st.success("🎉 飞行任务完成")

        # 修复：强制限制进度值 0~1，不会再报错
        progress = st.session_state.current_wp_idx / (len(st.session_state.flight_waypoints)-1)
        progress = min(progress, 1.0)
        st.progress(progress, text=f"任务进度：{round(progress*100,1)}%")

        col_map_flight, col_status = st.columns([2,1])
        with col_map_flight:
            st.subheader("🗺️ 实时飞行地图")
            m_flight = folium.Map(
                location=st.session_state.flight_waypoints[0],
                zoom_start=19,
                tiles="https://webst01.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}",
                attr="高德卫星地图"
            )
            for idx,obs in enumerate(st.session_state.polygon_memory):
                pts=obs["pts"]
                hh=obs["h"]
                if len(pts)>=3:
                    folium.Polygon(locations=pts,color="#dc2626",fill=True,fill_color="#dc2626",fill_opacity=0.45).add_to(m_flight)
            folium.PolyLine(st.session_state.flight_waypoints, color="#0066ff", weight=3, opacity=0.5).add_to(m_flight)
            flown_idx = int(st.session_state.current_wp_idx)
            flown_waypoints = st.session_state.flight_waypoints[:flown_idx+1]
            if len(flown_waypoints)>=2:
                folium.PolyLine(flown_waypoints, color="#22bb22", weight=4).add_to(m_flight)
            drone_pos = st.session_state.flight_waypoints[min(int(st.session_state.current_wp_idx), len(st.session_state.flight_waypoints)-1)]
            folium.CircleMarker(drone_pos, radius=10, color="orange", fill=True, fill_color="orange").add_to(m_flight)
            st_folium(m_flight, width="100%", height=500, key="flight_map")

        with col_status:
            st.subheader("📡 状态信息")
            st.success("✅ 地面站在线")
            st.success("✅ 飞控在线")
            st.info(f"📍 当前位置：{drone_pos[0]:.6f}, {drone_pos[1]:.6f}")
            st.info(f"🛫 飞行高度：{st.session_state.height} m")
            st.info(f"🚧 障碍物数量：{len(st.session_state.polygon_memory)} 个")

        if st.session_state.flight_running and not st.session_state.flight_paused:
            time.sleep(0.5)
            st.rerun()
