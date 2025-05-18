import asyncio
import sys
from pathlib import Path
import gc
import shutil
import json
from datetime import datetime, timedelta

# 添加项目根目录到 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

from services.monitoring_service import MonitoringService, MetricType

async def test_monitoring_service():
    # 创建测试目录
    test_dir = Path("test_metrics")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()
    
    try:
        # 初始化监控服务
        monitoring_service = MonitoringService(
            metrics_dir=str(test_dir),
            retention_days=1,
            collection_interval=1
        )
        
        # 测试记录自定义指标
        print("\n=== 测试记录自定义指标 ===")
        await monitoring_service.record_metric(
            name="test_metric",
            value=42.0,
            type=MetricType.CUSTOM,
            unit="count",
            labels={"test": "true"}
        )
        
        # 等待指标采集
        await asyncio.sleep(2)
        
        # 测试获取指标历史
        print("\n=== 测试获取指标历史 ===")
        history = await monitoring_service.get_metric_history("test_metric")
        print(f"历史数据点数量: {len(history)}")
        for value in history:
            print(f"值: {value.value}, 时间: {value.timestamp}")
        
        # 测试获取指标摘要
        print("\n=== 测试获取指标摘要 ===")
        summary = await monitoring_service.get_metric_summary("test_metric")
        print(f"指标摘要: {summary}")
        
        # 测试添加告警规则
        print("\n=== 测试添加告警规则 ===")
        rule = await monitoring_service.add_alert_rule(
            name="test_rule",
            metric_name="test_metric",
            condition=">",
            threshold=40.0,
            duration=60,
            severity="warning",
            description="测试告警规则"
        )
        print(f"创建的告警规则: {rule.dict()}")
        
        # 等待告警检查
        await asyncio.sleep(2)
        
        # 测试获取活跃告警
        print("\n=== 测试获取活跃告警 ===")
        alerts = await monitoring_service.get_active_alerts()
        print(f"活跃告警数量: {len(alerts)}")
        for alert in alerts:
            print(f"告警: {alert.dict()}")
        
        # 测试解决告警
        if alerts:
            print("\n=== 测试解决告警 ===")
            await monitoring_service.resolve_alert(0, "测试解决告警")
            alerts = await monitoring_service.get_active_alerts()
            print(f"解决后的活跃告警数量: {len(alerts)}")
        
        # 测试系统指标采集
        print("\n=== 测试系统指标采集 ===")
        await asyncio.sleep(2)  # 等待系统指标采集
        
        # 获取CPU使用率历史
        cpu_history = await monitoring_service.get_metric_history("cpu_usage")
        print(f"CPU使用率历史数据点数量: {len(cpu_history)}")
        if cpu_history:
            print(f"最新CPU使用率: {cpu_history[-1].value}%")
        
        # 获取内存使用率历史
        memory_history = await monitoring_service.get_metric_history("memory_usage")
        print(f"内存使用率历史数据点数量: {len(memory_history)}")
        if memory_history:
            print(f"最新内存使用率: {memory_history[-1].value}%")
        
        # 获取磁盘使用率历史
        disk_history = await monitoring_service.get_metric_history("disk_usage")
        print(f"磁盘使用率历史数据点数量: {len(disk_history)}")
        if disk_history:
            print(f"最新磁盘使用率: {disk_history[-1].value}%")
        
        # 测试时间范围查询
        print("\n=== 测试时间范围查询 ===")
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=1)
        recent_history = await monitoring_service.get_metric_history(
            "test_metric",
            start_time=start_time,
            end_time=end_time
        )
        print(f"时间范围内的数据点数量: {len(recent_history)}")
        
    finally:
        # 清理测试目录
        if test_dir.exists():
            shutil.rmtree(test_dir)

if __name__ == "__main__":
    asyncio.run(test_monitoring_service()) 