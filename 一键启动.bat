@echo off
chcp 65001
echo ======================================
echo 网络安全智能问答助手 启动中...
echo ======================================
echo.
echo 首次启动需要安装依赖，请耐心等待...
pip install -r requirements.txt -q
echo.
echo 启动成功！浏览器会自动打开...
echo 关闭这个窗口就会停止服务
echo.
start http://localhost:8501
streamlit run app.py
pause
