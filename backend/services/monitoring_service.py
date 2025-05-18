from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging
import psutil
import time
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
from collections import deque

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DateTimeEncoder(json.JSONEncoder):
    """处理 datetime 序列化的 JSON 编码器"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class MetricType(str, Enum):
    """指标类型枚举"""
    CPU = "cpu"               # CPU使用率
    MEMORY = "memory"         # 内存使用
    DISK = "disk"            # 磁盘使用
    NETWORK = "network"      # 网络使用
    PROCESS = "process"      # 进程信息
    CUSTOM = "custom"        # 自定义指标

class MetricValue(BaseModel):
    """指标值模型"""
    value: float = Field(..., description="指标值")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    labels: Dict[str, str] = Field(default_factory=dict, description="标签")

class Metric(BaseModel):
    """指标模型"""
    name: str = Field(..., description="指标名称")
    type: MetricType = Field(..., description="指标类型")
    description: str = Field(..., description="指标描述")
    values: List[MetricValue] = Field(default_factory=list, description="指标值历史")
    unit: str = Field(..., description="单位")
    labels: Dict[str, str] = Field(default_factory=dict, description="标签")

class AlertRule(BaseModel):
    """告警规则模型"""
    name: str = Field(..., description="规则名称")
    metric_name: str = Field(..., description="指标名称")
    condition: str = Field(..., description="告警条件")
    threshold: float = Field(..., description="阈值")
    duration: int = Field(..., description="持续时间(秒)")
    severity: str = Field(..., description="严重程度")
    description: str = Field(..., description="规则描述")

class Alert(BaseModel):
    """告警模型"""
    rule_name: str = Field(..., description="规则名称")
    metric_name: str = Field(..., description="指标名称")
    value: float = Field(..., description="触发值")
    threshold: float = Field(..., description="阈值")
    timestamp: datetime = Field(default_factory=datetime.now, description="触发时间")
    status: str = Field(default="active", description="告警状态")
    severity: str = Field(..., description="严重程度")

class MonitoringService:
    """性能监控服务"""
    
    def __init__(
        self,
        metrics_dir: str = "metrics",
        retention_days: int = 7,
        collection_interval: int = 60
    ):
        """
        初始化监控服务
        
        Args:
            metrics_dir: 指标存储目录
            retention_days: 数据保留天数
            collection_interval: 采集间隔(秒)
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self.collection_interval = collection_interval
        self.metrics: Dict[str, Metric] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alerts: List[Alert] = []
        self._load_metrics()
        self._load_alert_rules()
        self._start_collection()
    
    def _load_metrics(self):
        """加载指标配置"""
        metrics_file = self.metrics_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics_dict = json.load(f)
                for name, metric_dict in metrics_dict.items():
                    self.metrics[name] = Metric(**metric_dict)
    
    def _save_metrics(self):
        """保存指标配置"""
        metrics_file = self.metrics_dir / "metrics.json"
        metrics_dict = {name: metric.dict() for name, metric in self.metrics.items()}
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, ensure_ascii=False, indent=2, cls=DateTimeEncoder)
    
    def _load_alert_rules(self):
        """加载告警规则"""
        rules_file = self.metrics_dir / "alert_rules.json"
        if rules_file.exists():
            with open(rules_file, "r", encoding="utf-8") as f:
                rules_dict = json.load(f)
                for name, rule_dict in rules_dict.items():
                    self.alert_rules[name] = AlertRule(**rule_dict)
    
    def _save_alert_rules(self):
        """保存告警规则"""
        rules_file = self.metrics_dir / "alert_rules.json"
        rules_dict = {name: rule.dict() for name, rule in self.alert_rules.items()}
        with open(rules_file, "w", encoding="utf-8") as f:
            json.dump(rules_dict, f, ensure_ascii=False, indent=2, cls=DateTimeEncoder)
    
    def _start_collection(self):
        """启动指标采集"""
        asyncio.create_task(self._collect_metrics())
    
    async def _collect_metrics(self):
        """采集指标"""
        while True:
            try:
                # 采集系统指标
                await self._collect_system_metrics()
                
                # 检查告警规则
                await self._check_alert_rules()
                
                # 清理过期数据
                await self._cleanup_old_data()
                
                # 等待下一次采集
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"指标采集失败: {str(e)}")
                await asyncio.sleep(5)  # 发生错误时等待5秒后重试
    
    async def _collect_system_metrics(self):
        """采集系统指标"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        await self.record_metric(
            name="cpu_usage",
            value=cpu_percent,
            type=MetricType.CPU,
            unit="%"
        )
        
        # 内存使用
        memory = psutil.virtual_memory()
        await self.record_metric(
            name="memory_usage",
            value=memory.percent,
            type=MetricType.MEMORY,
            unit="%"
        )
        
        # 磁盘使用
        disk = psutil.disk_usage('/')
        await self.record_metric(
            name="disk_usage",
            value=disk.percent,
            type=MetricType.DISK,
            unit="%"
        )
        
        # 网络使用
        net_io = psutil.net_io_counters()
        await self.record_metric(
            name="network_bytes_sent",
            value=net_io.bytes_sent,
            type=MetricType.NETWORK,
            unit="bytes"
        )
        await self.record_metric(
            name="network_bytes_recv",
            value=net_io.bytes_recv,
            type=MetricType.NETWORK,
            unit="bytes"
        )
        
        # 进程信息
        process = psutil.Process()
        await self.record_metric(
            name="process_cpu_usage",
            value=process.cpu_percent(),
            type=MetricType.PROCESS,
            unit="%"
        )
        await self.record_metric(
            name="process_memory_usage",
            value=process.memory_percent(),
            type=MetricType.PROCESS,
            unit="%"
        )
    
    async def record_metric(
        self,
        name: str,
        value: float,
        type: MetricType,
        unit: str = "",
        labels: Optional[Dict[str, str]] = None
    ):
        """
        记录指标值
        
        Args:
            name: 指标名称
            value: 指标值
            type: 指标类型
            unit: 单位
            labels: 标签
        """
        try:
            # 获取或创建指标
            if name not in self.metrics:
                self.metrics[name] = Metric(
                    name=name,
                    type=type,
                    description=f"Metric {name}",
                    unit=unit,
                    labels=labels or {}
                )
            
            # 添加指标值
            metric_value = MetricValue(
                value=value,
                labels=labels or {}
            )
            self.metrics[name].values.append(metric_value)
            
            # 保存指标配置
            self._save_metrics()
            
        except Exception as e:
            logger.error(f"记录指标失败: {str(e)}")
            raise
    
    async def _check_alert_rules(self):
        """检查告警规则"""
        for rule_name, rule in self.alert_rules.items():
            if rule.metric_name not in self.metrics:
                continue
            
            metric = self.metrics[rule.metric_name]
            if not metric.values:
                continue
            
            # 获取最近的值
            recent_values = [
                v.value for v in metric.values
                if (datetime.now() - v.timestamp).total_seconds() <= rule.duration
            ]
            
            if not recent_values:
                continue
            
            # 计算平均值
            avg_value = sum(recent_values) / len(recent_values)
            
            # 检查是否触发告警
            if eval(f"{avg_value} {rule.condition} {rule.threshold}"):
                # 创建告警
                alert = Alert(
                    rule_name=rule_name,
                    metric_name=rule.metric_name,
                    value=avg_value,
                    threshold=rule.threshold,
                    severity=rule.severity
                )
                self.alerts.append(alert)
                logger.warning(f"触发告警: {alert.dict()}")
    
    async def _cleanup_old_data(self):
        """清理过期数据"""
        cutoff_time = datetime.now() - timedelta(days=self.retention_days)
        
        for metric in self.metrics.values():
            metric.values = [
                v for v in metric.values
                if v.timestamp > cutoff_time
            ]
        
        self.alerts = [
            alert for alert in self.alerts
            if alert.timestamp > cutoff_time
        ]
    
    async def add_alert_rule(
        self,
        name: str,
        metric_name: str,
        condition: str,
        threshold: float,
        duration: int,
        severity: str,
        description: str
    ) -> AlertRule:
        """
        添加告警规则
        
        Args:
            name: 规则名称
            metric_name: 指标名称
            condition: 告警条件
            threshold: 阈值
            duration: 持续时间(秒)
            severity: 严重程度
            description: 规则描述
            
        Returns:
            AlertRule: 创建的告警规则
        """
        try:
            rule = AlertRule(
                name=name,
                metric_name=metric_name,
                condition=condition,
                threshold=threshold,
                duration=duration,
                severity=severity,
                description=description
            )
            
            self.alert_rules[name] = rule
            self._save_alert_rules()
            
            return rule
            
        except Exception as e:
            logger.error(f"添加告警规则失败: {str(e)}")
            raise
    
    async def get_metric_history(
        self,
        metric_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[MetricValue]:
        """
        获取指标历史数据
        
        Args:
            metric_name: 指标名称
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            List[MetricValue]: 指标值列表
        """
        try:
            if metric_name not in self.metrics:
                return []
            
            metric = self.metrics[metric_name]
            values = metric.values
            
            if start_time:
                values = [v for v in values if v.timestamp >= start_time]
            if end_time:
                values = [v for v in values if v.timestamp <= end_time]
            
            return values
            
        except Exception as e:
            logger.error(f"获取指标历史失败: {str(e)}")
            raise
    
    async def get_active_alerts(self) -> List[Alert]:
        """
        获取活跃告警
        
        Returns:
            List[Alert]: 告警列表
        """
        return [alert for alert in self.alerts if alert.status == "active"]
    
    async def resolve_alert(self, alert_index: int, comment: Optional[str] = None):
        """
        解决告警
        
        Args:
            alert_index: 告警索引
            comment: 解决说明
        """
        try:
            if 0 <= alert_index < len(self.alerts):
                self.alerts[alert_index].status = "resolved"
                logger.info(f"解决告警: {self.alerts[alert_index].dict()}")
                
        except Exception as e:
            logger.error(f"解决告警失败: {str(e)}")
            raise
    
    async def get_metric_summary(self, metric_name: str) -> Dict[str, float]:
        """
        获取指标统计摘要
        
        Args:
            metric_name: 指标名称
            
        Returns:
            Dict[str, float]: 统计摘要
        """
        try:
            if metric_name not in self.metrics:
                return {}
            
            values = [v.value for v in self.metrics[metric_name].values]
            if not values:
                return {}
            
            return {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "latest": values[-1]
            }
            
        except Exception as e:
            logger.error(f"获取指标摘要失败: {str(e)}")
            raise 