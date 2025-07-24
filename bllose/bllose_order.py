import ccxt
import time
import os
from dotenv import load_dotenv

def create_virtual_sell_order(api_key, secret_key, passphrase, symbol, amount, price=None):
    """
    在OKX虚拟交易环境创建卖出订单
    
    参数:
    api_key (str): OKX API Key
    secret_key (str): OKX Secret Key
    passphrase (str): OKX Passphrase
    symbol (str): 交易对，如 'BTC-USDT'
    amount (float): 卖出数量
    price (float, optional): 卖出价格，默认为None表示市价单
    
    返回:
    dict: 订单信息

    十二、接口报错：50101 APIKey does not match current environment
    这个是因为 APIKey 和当前环境不匹配导致的，
    实盘调用需要使用实盘 APIKey ，且请求的header 里面 x-simulated-trading这个参数值需要为 0 ；
    模拟盘调用需要使用模拟盘 APIKey ，且请求的 header 里面 x-simulated-trading 这个参数值需要为 1 。
    """
    # 初始化OKX交易所对象，设置为测试环境
    exchange = ccxt.okx({
        'apiKey': api_key,
        'secret': secret_key,
        'password': passphrase,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',  # 默认为现货交易
            'test': True,           # 启用测试模式（虚拟交易）
        },
    })
    exchange.https_proxy = 'http://127.0.0.1:7897/'
    exchange.simulate = True
    # exchange.UpdateProxySettings()
    
    try:
        # 检查账户余额
        balance = exchange.fetch_balance()
        if symbol.split('-')[0] in balance['total']:
            print(f"当前{symbol.split('-')[0]}余额: {balance['total'][symbol.split('-')[0]]}")
        else:
            print(f"当前账户没有{symbol.split('-')[0]}资产")
            return None
        
        # 创建卖出订单
        if price:
            # 限价卖出
            order = exchange.create_limit_sell_order(symbol, amount, price)
            print(f"限价卖出订单创建成功: {order['id']}")
        else:
            # 市价卖出
            order = exchange.create_market_sell_order(symbol, amount)
            print(f"市价卖出订单创建成功: {order['id']}")
        
        # 等待一段时间后查询订单状态
        time.sleep(2)
        order_status = exchange.fetch_order(order['id'], symbol)
        print(f"订单状态: {order_status['status']}")
        
        return order
    
    except ccxt.NetworkError as e:
        print(f"网络错误: {e}")
    except ccxt.ExchangeError as e:
        print(f"交易所错误: {e}")
    except Exception as e:
        print(f"未知错误: {e}")
    return None

if __name__ == "__main__":
    load_dotenv()

    # 填入你的OKX API凭证 (请使用测试环境的API Key)
    API_KEY = os.getenv('API_KEY')
    SECRET_KEY = os.getenv('SECRET_KEY')
    PASSPHRASE = os.getenv('PASSPHRASE')
    
    # 交易参数
    SYMBOL = 'BTC-USDT'  # 交易对
    AMOUNT = 0.001       # 卖出数量
    PRICE = 120000        # 卖出价格，设为None表示市价单
    
    # 创建虚拟卖出订单
    order = create_virtual_sell_order(API_KEY, SECRET_KEY, PASSPHRASE, SYMBOL, AMOUNT, PRICE)
    
    if order:
        print("订单详情:")
        print(f"交易对: {order['symbol']}")
        print(f"类型: {order['type']}")
        print(f"方向: {order['side']}")
        print(f"数量: {order['amount']}")
        if order['price']:
            print(f"价格: {order['price']}")
        print(f"状态: {order['status']}")    