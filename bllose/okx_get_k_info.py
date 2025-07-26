import ccxt
import pandas as pd
import time
import logging
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('OKX-Kline-Fetcher')

class OKXKlineFetcher:
    def __init__(self, test_mode=False):
        """
        初始化 OKX K线数据获取器
        
        Args:
            test_mode: 是否使用测试环境
        """
        # 创建 OKX 交易所实例
        self.exchange = ccxt.okx({
            'enableRateLimit': True,
        })
        self.exchange.https_proxy = 'http://127.0.0.1:7897/'
        self.exchange.simulate = True  # 启用模拟交易模式
        
        # 设置为测试模式（沙盒环境）
        if test_mode:
            self.exchange.set_sandbox_mode(True)
            logger.info("使用 OKX 测试环境")
        else:
            logger.info("使用 OKX 生产环境")
        
        # 验证 API 连接
        self._verify_connection()
        
        # 定义时间周期映射
        self.timeframe_mapping = {
            '1m': '1m',
            '3m': '3m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1H',
            '2h': '2H',
            '4h': '4H',
            '6h': '6H',
            '12h': '12H',
            '1d': '1D',
            '1w': '1W',
            '1M': '1M',  # OKX 使用 1M 表示 1 个月
        }
    
    def _verify_connection(self):
        """验证与 OKX 交易所的连接"""
        try:
            markets = self.exchange.load_markets()
            logger.info(f"连接成功，加载了 {len(markets)} 个交易对")
            return True
        except Exception as e:
            logger.error(f"连接失败: {e}")
            return False
    
    def fetch_ohlcv(self, symbol='BTC/USDT', timeframe='5m', limit=100, since=None):
        """
        获取指定交易对的 K 线数据
        
        Args:
            symbol: 交易对，如 'BTC/USDT'
            timeframe: 时间周期，如 '1m', '5m', '1h' 等
            limit: 获取的 K 线数量上限
            since: 开始时间戳（毫秒）
            
        Returns:
            pandas.DataFrame: 包含 K 线数据的 DataFrame
        """
        try:
            # 转换时间周期为 OKX 支持的格式
            okx_timeframe = self.timeframe_mapping.get(timeframe, timeframe)
            
            # 获取 K 线数据
            ohlcv = self.exchange.fetch_ohlcv(symbol, okx_timeframe, since, limit)
            
            if not ohlcv:
                logger.warning(f"未获取到 {symbol} {timeframe} 的 K 线数据")
                return pd.DataFrame()
            
            # 转换为 DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 转换时间戳为 datetime 格式
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 设置 datetime 为索引
            df.set_index('datetime', inplace=True)
            
            logger.info(f"成功获取 {len(df)} 条 {symbol} {timeframe} K 线数据")
            return df
        except Exception as e:
            logger.error(f"获取 K 线数据失败: {e}")
            return pd.DataFrame()
    
    def fetch_ohlcv_by_days(self, symbol='BTC/USDT', timeframe='5m', days=1):
        """
        获取最近指定天数的 K 线数据
        
        Args:
            symbol: 交易对
            timeframe: 时间周期
            days: 天数
            
        Returns:
            pandas.DataFrame: 包含 K 线数据的 DataFrame
        """
        # 计算开始时间
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # 转换为毫秒时间戳
        since = int(start_time.timestamp() * 1000)
        
        # 计算需要获取的 K 线数量
        timeframe_minutes = int(timeframe.replace('m', '')) if 'm' in timeframe else \
                           int(timeframe.replace('h', '')) * 60 if 'h' in timeframe else \
                           24 * 60 if timeframe == '1d' else 7 * 24 * 60
        
        limit = int(days * 24 * 60 / timeframe_minutes) + 10  # 增加一些余量
        
        return self.fetch_ohlcv(symbol, timeframe, limit, since)
    
    def fetch_ohlcv_history(self, symbol='BTC/USDT', timeframe='5m', start_date=None, end_date=None):
        """
        获取历史 K 线数据（支持分页获取）
        
        Args:
            symbol: 交易对
            timeframe: 时间周期
            start_date: 开始日期，格式为 'YYYY-MM-DD'
            end_date: 结束日期，格式为 'YYYY-MM-DD'
            
        Returns:
            pandas.DataFrame: 包含 K 线数据的 DataFrame
        """
        if not start_date:
            # 默认获取最近 30 天数据
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
        else:
            # 解析日期字符串
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            if end_date:
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
            else:
                end_date = datetime.now()
        
        # 转换为毫秒时间戳
        since = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)
        
        # 每个请求的最大 K 线数量
        limit = 300
        all_ohlcv = []
        
        while since < end_timestamp:
            logger.info(f"从 {datetime.fromtimestamp(since/1000)} 开始获取数据")
            
            # 获取数据
            ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe_mapping[timeframe], since, limit)
            
            if not ohlcv:
                logger.warning("未获取到更多数据")
                break
                
            # 添加到结果列表
            all_ohlcv.extend(ohlcv)
            
            # 更新 since 为最后一条数据的时间戳 + 1 个时间周期
            last_timestamp = ohlcv[-1][0]
            timeframe_seconds = {
                '1m': 60,
                '3m': 180,
                '5m': 300,
                '15m': 900,
                '30m': 1800,
                '1h': 3600,
                '4h': 14400,
                '1d': 86400,
            }.get(timeframe, 300)  # 默认 5 分钟
            
            since = last_timestamp + timeframe_seconds * 1000
            
            # 避免触发 API 速率限制
            time.sleep(self.exchange.rateLimit / 1000)
        
        if not all_ohlcv:
            logger.warning(f"未获取到 {symbol} {timeframe} 的任何历史 K 线数据")
            return pd.DataFrame()
            
        # 转换为 DataFrame
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        logger.info(f"成功获取 {len(df)} 条 {symbol} {timeframe} 历史 K 线数据")
        return df

def main():
    # 创建 K 线数据获取器实例
    fetcher = OKXKlineFetcher(test_mode=False)
    
    # 交易对和时间周期
    symbol = 'BTC/USDT'
    timeframe = '5m'
    
    # 方法1: 获取最近 100 条 K 线数据
    logger.info("获取最近 100 条 K 线数据...")
    recent_ohlcv = fetcher.fetch_ohlcv(symbol, timeframe, limit=100)
    if not recent_ohlcv.empty:
        print("\n最近 10 条 K 线数据:")
        print(recent_ohlcv.tail(10))
    
    # 方法2: 获取最近 3 天的 K 线数据
    logger.info("\n获取最近 3 天的 K 线数据...")
    three_days_ohlcv = fetcher.fetch_ohlcv_by_days(symbol, timeframe, days=3)
    if not three_days_ohlcv.empty:
        print(f"\n{three_days_ohlcv.shape[0]} 条 K 线数据:")
        print(three_days_ohlcv.describe())
    
    # 方法3: 获取指定日期范围内的历史 K 线数据
    logger.info("\n获取 2023-01-01 至 2023-01-07 的 K 线数据...")
    history_ohlcv = fetcher.fetch_ohlcv_history(symbol, timeframe, '2023-01-01', '2023-01-07')
    if not history_ohlcv.empty:
        print(f"\n{history_ohlcv.shape[0]} 条历史 K 线数据:")
        print(f"数据范围: {history_ohlcv.index.min()} 至 {history_ohlcv.index.max()}")

if __name__ == "__main__":
    main()    