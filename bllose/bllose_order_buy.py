import ccxt
import os
import time
import logging
from pprint import pprint
from dotenv import load_dotenv

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('OKX-Trading-Bot')

class OKXTrader:
    def __init__(self, api_key, api_secret, passphrase, test_mode=True):
        """
        初始化 OKX 交易接口
        
        Args:
            api_key: OKX API Key
            api_secret: OKX API Secret
            passphrase: OKX API Passphrase
            test_mode: 是否使用测试环境
        """
        # 创建 OKX 交易所实例
        self.exchange = ccxt.okx({
            'apiKey': api_key,
            'secret': api_secret,
            'password': passphrase,
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
        self._verify_credentials()
    
    def _verify_credentials(self):
        """验证 API 凭证是否有效"""
        try:
            balance = self.exchange.fetch_balance()
            logger.info("API 验证成功")
            return balance
        except Exception as e:
            logger.error(f"API 验证失败: {e}")
            raise
    
    def get_market_price(self, symbol):
        """
        获取指定交易对的当前市场价格
        
        Args:
            symbol: 交易对符号，如 'BTC/USDT'
        
        Returns:
            dict: 包含买价和卖价的字典
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                'bid': ticker['bid'],  # 买价
                'ask': ticker['ask'],  # 卖价
                'last': ticker['last']  # 最新价格
            }
        except Exception as e:
            logger.error(f"获取市场价格失败: {e}")
            return None
    
    def get_balance(self, currency=None):
        """
        获取账户余额
        
        Args:
            currency: 可选，指定货币类型，如 'USDT'
        
        Returns:
            float: 指定货币的可用余额，或全部余额字典
        """
        try:
            balance = self.exchange.fetch_balance()
            if currency:
                return balance['free'].get(currency, 0.0)
            else:
                return balance
        except Exception as e:
            logger.error(f"获取账户余额失败: {e}")
            return None
    
    def create_limit_buy_order(self, symbol, amount, price):
        """
        创建限价买入订单
        
        Args:
            symbol: 交易对符号，如 'BTC/USDT'
            amount: 买入数量
            price: 买入价格
        
        Returns:
            dict: 订单信息
        """
        try:
            # 检查余额是否充足
            quote_currency = symbol.split('/')[1]  # 获取计价货币
            available_balance = self.get_balance(quote_currency)
            required_amount = amount * price
            
            if available_balance < required_amount:
                logger.error(f"余额不足: 需要 {required_amount} {quote_currency}, 可用 {available_balance} {quote_currency}")
                return None
            
            # 创建限价买入订单
            order = self.exchange.create_limit_buy_order(symbol, amount, price)
            logger.info(f"限价买入订单创建成功: {symbol}, 数量: {amount}, 价格: {price}")
            logger.info(f"订单 ID: {order['id']}")
            return order
        except Exception as e:
            logger.error(f"创建限价买入订单失败: {e}")
            return None
    
    def check_order_status(self, symbol, order_id):
        """
        检查订单状态
        
        Args:
            symbol: 交易对符号
            order_id: 订单 ID
        
        Returns:
            dict: 订单状态信息
        """
        try:
            order = self.exchange.fetch_order(order_id, symbol)
            return order
        except Exception as e:
            logger.error(f"检查订单状态失败: {e}")
            return None

def main():
    load_dotenv()

    # 替换为你的 OKX API 凭证
    API_KEY = os.getenv('API_KEY')
    API_SECRET = os.getenv('SECRET_KEY')
    PASSPHRASE = os.getenv('PASSPHRASE')
    
    # 交易参数
    SYMBOL = 'BTC/USDT'  # 交易对
    AMOUNT = 0.001       # 买入数量
    PRICE = 20000        # 买入价格（根据市场情况调整）
    TEST_MODE = True     # 是否使用测试环境
    
    # 创建交易实例
    trader = OKXTrader(API_KEY, API_SECRET, PASSPHRASE, TEST_MODE)
    
    # 获取当前市场价格
    market_price = trader.get_market_price(SYMBOL)
    if market_price:
        logger.info(f"当前市场价格: 买价={market_price['bid']}, 卖价={market_price['ask']}, 最新价={market_price['last']}")
        
        # 显示账户余额
        base_currency = SYMBOL.split('/')[0]
        quote_currency = SYMBOL.split('/')[1]
        
        base_balance = trader.get_balance(base_currency)
        quote_balance = trader.get_balance(quote_currency)
        
        logger.info(f"账户余额: {base_currency}={base_balance}, {quote_currency}={quote_balance}")
        
        # 创建限价买入订单
        order = trader.create_limit_buy_order(SYMBOL, AMOUNT, PRICE)
        
        if order:
            # 检查订单状态
            time.sleep(2)  # 等待2秒，确保订单已被处理
            order_status = trader.check_order_status(SYMBOL, order['id'])
            
            if order_status:
                logger.info(f"订单状态: {order_status['status']}")
                pprint(order_status)

if __name__ == "__main__":
    main()    