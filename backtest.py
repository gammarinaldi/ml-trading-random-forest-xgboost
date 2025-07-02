import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use('Agg')

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class Backtester:
    """
    Minimal backtesting system to identify and fix the signal generation issue
    """
    
    def __init__(self, model_path='models/unified_price_regressor.pkl', initial_capital=10000):
        self.model = self.load_model(model_path)
        self.initial_capital = initial_capital
        
        # Conservative optimized parameters for better performance
        self.spread_pips = 1.5
        self.slippage_pips = 0.3
        self.max_position_size = 0.005  # 0.5% risk per trade (very conservative)
        self.min_confidence = 0.0008    # Higher confidence threshold
        
        # Improved risk management
        self.stop_loss_pips = 18        # Wider stop loss
        self.take_profit_pips = 54      # 3:1 risk/reward ratio
        
        self.trades = []
        self.equity_curve = []
        
    def load_model(self, model_path):
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"✅ Model loaded successfully")
            return model
        else:
            raise FileNotFoundError(f"❌ Model not found at {model_path}")
    
    def add_technical_indicators(self, df, price_col='close'):
        """Add minimal required technical indicators"""
        df['sma_5'] = df[price_col].rolling(window=5).mean()
        df['sma_10'] = df[price_col].rolling(window=10).mean()
        df['sma_20'] = df[price_col].rolling(window=20).mean()
        df['sma_50'] = df[price_col].rolling(window=50).mean()
        
        df['ema_5'] = df[price_col].ewm(span=5).mean()
        df['ema_10'] = df[price_col].ewm(span=10).mean()
        df['ema_20'] = df[price_col].ewm(span=20).mean()
        
        df['price_vs_sma5'] = df[price_col] / df['sma_5'] - 1
        df['price_vs_sma10'] = df[price_col] / df['sma_10'] - 1
        df['price_vs_sma20'] = df[price_col] / df['sma_20'] - 1
        
        df['sma5_vs_sma10'] = df['sma_5'] / df['sma_10'] - 1
        df['sma10_vs_sma20'] = df['sma_10'] / df['sma_20'] - 1
        
        df['returns'] = df[price_col].pct_change()
        df['volatility_5'] = df['returns'].rolling(window=5).std()
        df['volatility_10'] = df['returns'].rolling(window=10).std()
        df['volatility_20'] = df[price_col].rolling(window=20).std()
        
        df['momentum_3'] = df[price_col] / df[price_col].shift(3) - 1
        df['momentum_5'] = df[price_col] / df[price_col].shift(5) - 1
        df['momentum_10'] = df[price_col] / df[price_col].shift(10) - 1
        
        df['roc_5'] = df[price_col].pct_change(5)
        df['roc_10'] = df[price_col].pct_change(10)
        
        # Bollinger Bands
        df['bb_middle'] = df[price_col].rolling(window=20).mean()
        bb_std = df[price_col].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df[price_col] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # RSI
        delta = df[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        df['high_5'] = df[price_col].rolling(window=5).max()
        df['low_5'] = df[price_col].rolling(window=5).min()
        df['high_10'] = df[price_col].rolling(window=10).max()
        df['low_10'] = df[price_col].rolling(window=10).min()
        
        df['dist_from_high5'] = (df['high_5'] - df[price_col]) / df[price_col]
        df['dist_from_low5'] = (df[price_col] - df['low_5']) / df[price_col]
        
        return df
    
    def predict_price_and_direction(self, recent_data, feature_cols, n_steps=100):
        """Make prediction - debug version"""
        if len(recent_data) < n_steps:
            return None, None, 0.0
        
        try:
            features = recent_data[feature_cols].iloc[-n_steps:].values.flatten()
            features = features.reshape(1, -1)
            
            # Check for NaN values
            if np.isnan(features).any():
                return None, None, 0.0
            
            price_pred = self.model.predict(features)[0]
            current_price = recent_data['close'].iloc[-1]
            price_change = (price_pred - current_price) / current_price
            
            direction_pred = 1 if price_change > 0 else 0
            confidence = abs(price_change)
            
            return direction_pred, price_pred, confidence
        except Exception as e:
            print(f"  Prediction error: {e}")
            return None, None, 0.0
    
    def calculate_position_size(self, confidence, account_balance):
        """Simple position sizing"""
        risk_amount = account_balance * self.max_position_size
        pip_value = 0.0001
        stop_loss_amount = self.stop_loss_pips * pip_value
        lot_size = risk_amount / (stop_loss_amount * 100000)
        return max(0.01, min(lot_size, 0.5))
    
    def execute_trade(self, signal, predicted_price, confidence, current_price, account_balance, timestamp):
        """Execute simple trade"""
        lot_size = self.calculate_position_size(confidence, account_balance)
        
        pip_value = 0.0001
        if signal == 1:  # BUY
            entry_price = current_price + (self.spread_pips / 2) * pip_value
            stop_loss = entry_price - self.stop_loss_pips * pip_value
            take_profit = entry_price + self.take_profit_pips * pip_value
        else:  # SELL
            entry_price = current_price - (self.spread_pips / 2) * pip_value
            stop_loss = entry_price + self.stop_loss_pips * pip_value
            take_profit = entry_price - self.take_profit_pips * pip_value
        
        transaction_cost = (self.spread_pips + self.slippage_pips) * pip_value * lot_size * 100000
        
        trade = {
            'timestamp': timestamp,
            'signal': 'BUY' if signal == 1 else 'SELL',
            'entry_price': entry_price,
            'predicted_price': predicted_price,
            'lot_size': lot_size,
            'confidence': confidence,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'transaction_cost': transaction_cost,
            'status': 'OPEN'
        }
        
        return trade
    
    def update_open_trades(self, current_price, timestamp):
        """Update trades"""
        closed_trades = []
        
        for trade in self.trades:
            if trade['status'] == 'OPEN':
                if trade['signal'] == 'BUY':
                    if current_price <= trade['stop_loss']:
                        trade['exit_price'] = trade['stop_loss']
                        trade['exit_reason'] = 'STOP_LOSS'
                        trade['status'] = 'CLOSED'
                        trade['exit_timestamp'] = timestamp
                    elif current_price >= trade['take_profit']:
                        trade['exit_price'] = trade['take_profit']
                        trade['exit_reason'] = 'TAKE_PROFIT'
                        trade['status'] = 'CLOSED'
                        trade['exit_timestamp'] = timestamp
                else:  # SELL
                    if current_price >= trade['stop_loss']:
                        trade['exit_price'] = trade['stop_loss']
                        trade['exit_reason'] = 'STOP_LOSS'
                        trade['status'] = 'CLOSED'
                        trade['exit_timestamp'] = timestamp
                    elif current_price <= trade['take_profit']:
                        trade['exit_price'] = trade['take_profit']
                        trade['exit_reason'] = 'TAKE_PROFIT'
                        trade['status'] = 'CLOSED'
                        trade['exit_timestamp'] = timestamp
                
                if trade['status'] == 'CLOSED':
                    closed_trades.append(trade)
        
        return closed_trades
    
    def calculate_trade_pnl(self, trade):
        """Calculate P&L"""
        if trade['status'] != 'CLOSED':
            return 0
        
        lot_size = trade['lot_size']
        contract_size = 100000
        
        if trade['signal'] == 'BUY':
            pnl = (trade['exit_price'] - trade['entry_price']) * lot_size * contract_size
        else:  # SELL
            pnl = (trade['entry_price'] - trade['exit_price']) * lot_size * contract_size
        
        pnl -= trade['transaction_cost']
        return pnl
    
    def run_simple_backtest(self, data, n_steps=100, start_date=None, debug_mode=True):
        """Run simple backtest with debugging"""
        print("🚀 Starting simple backtest with minimal filtering...")
        
        data = data.copy()
        data = self.add_technical_indicators(data, 'close')
        data = data.dropna()
        
        if start_date:
            data = data[data.index >= start_date]
        
        print(f"📊 Simple backtesting period: {data.index[0]} to {data.index[-1]}")
        print(f"📈 Total periods: {len(data)}")
        
        feature_cols = [col for col in data.columns if col not in ['close']]
        print(f"📊 Features: {len(feature_cols)}")
        
        account_balance = self.initial_capital
        self.trades = []
        self.equity_curve = []
        
        total_signals = 0
        valid_predictions = 0
        confidence_above_threshold = 0
        
        # Test periods - full dataset when not debugging
        test_periods = min(1000, len(data) - n_steps) if debug_mode else len(data) - n_steps
        
        print(f"🔍 Testing {test_periods} periods...")
        
        with tqdm(total=test_periods, desc="Simple Backtesting", unit="periods") as pbar:
            for i in range(n_steps, n_steps + test_periods):
                current_timestamp = data.index[i]
                current_price = data['close'].iloc[i]
                recent_data = data.iloc[i-n_steps:i]
                
                # Update open trades
                closed_trades = self.update_open_trades(current_price, current_timestamp)
                
                # Process closed trades
                for trade in closed_trades:
                    pnl = self.calculate_trade_pnl(trade)
                    account_balance += pnl
                    trade['pnl'] = pnl
                
                # Make prediction
                signal, predicted_price, confidence = self.predict_price_and_direction(
                    recent_data, feature_cols, n_steps
                )
                
                if signal is not None:
                    total_signals += 1
                    
                    if confidence >= self.min_confidence:
                        confidence_above_threshold += 1
                        
                        # Add simple trend filter
                        current_row = recent_data.iloc[-1]
                        sma_5 = current_row.get('sma_5', 0)
                        sma_20 = current_row.get('sma_20', 0)
                        
                        trend_ok = True
                        if pd.notna(sma_5) and pd.notna(sma_20):
                            uptrend = sma_5 > sma_20
                            if signal == 1 and not uptrend:  # BUY but downtrend
                                trend_ok = False
                            if signal == 0 and uptrend:     # SELL but uptrend
                                trend_ok = False
                        
                        if trend_ok:  # Only execute if trend agrees
                            open_trades = [t for t in self.trades if t['status'] == 'OPEN']
                            if len(open_trades) == 0:  # Only 1 trade at a time
                                new_trade = self.execute_trade(
                                    signal, predicted_price, confidence, current_price,
                                    account_balance, current_timestamp
                                )
                                
                                if new_trade:
                                    self.trades.append(new_trade)
                                    account_balance -= new_trade['transaction_cost']
                                    
                                    if debug_mode and len(self.trades) <= 5:
                                        print(f"  📈 Trade {len(self.trades)}: {new_trade['signal']} at {current_price:.5f}, confidence: {confidence:.6f}")
                
                # Record equity
                self.equity_curve.append({
                    'timestamp': current_timestamp,
                    'balance': account_balance,
                    'price': current_price
                })
                
                pbar.update(1)
        
        # Close remaining trades
        final_price = data['close'].iloc[n_steps + test_periods - 1]
        final_timestamp = data.index[n_steps + test_periods - 1]
        
        for trade in self.trades:
            if trade['status'] == 'OPEN':
                trade['exit_price'] = final_price
                trade['exit_reason'] = 'END_OF_DATA'
                trade['status'] = 'CLOSED'
                trade['exit_timestamp'] = final_timestamp
                pnl = self.calculate_trade_pnl(trade)
                account_balance += pnl
                trade['pnl'] = pnl
        
        closed_trades = [t for t in self.trades if t['status'] == 'CLOSED']
        
        print(f"✅ Simple backtest completed!")
        print(f"📊 Total signals: {total_signals}")
        print(f"📊 Valid predictions: {valid_predictions}")
        print(f"📊 Confidence above threshold: {confidence_above_threshold}")
        print(f"📈 Executed trades: {len(closed_trades)}")
        print(f"💰 Final balance: ${account_balance:,.2f}")
        print(f"📊 Total return: {((account_balance - self.initial_capital) / self.initial_capital * 100):+.2f}%")
        
        return account_balance, len(closed_trades)

def main():
    """Run the simple working backtesting system"""
    print("🚀 Optimized Forex Trading System - Final Backtest\n")
    
    csv_file = 'EURUSDm_H1_201801020600_202412310000.csv'
    
    if not os.path.exists(csv_file):
        print(f"❌ Data file {csv_file} not found!")
        return
    
    try:
        # Load data
        print(f"📈 Loading historical data from {csv_file}")
        df = pd.read_csv(csv_file, sep='\t')
        
        if '<DATE>' in df.columns and '<TIME>' in df.columns and '<CLOSE>' in df.columns:
            df['datetime'] = df['<DATE>'] + ' ' + df['<TIME>']
            data = pd.DataFrame()
            data['time'] = df['datetime']
            data['close'] = df['<CLOSE>']
            data['time'] = pd.to_datetime(data['time'], format='%Y.%m.%d %H:%M:%S')
            data = data.set_index('time')
            data['close'] = pd.to_numeric(data['close'], errors='coerce')
            data = data.dropna()
            
            print(f"✅ Loaded {len(data)} price records")
        else:
            raise ValueError("Unsupported data format")
        
        # Initialize simple backtester
        simple_backtester = Backtester(
            model_path='models/unified_price_regressor.pkl',
            initial_capital=10000
        )
        
        print(f"\n🔧 SIMPLE PARAMETERS:")
        print(f"  💰 Min confidence: {simple_backtester.min_confidence:.4f}")
        print(f"  📏 Risk per trade: {simple_backtester.max_position_size*100:.1f}%")
        print(f"  🛡️ Stop loss: {simple_backtester.stop_loss_pips} pips")
        print(f"  🎯 Take profit: {simple_backtester.take_profit_pips} pips")
        
        # Run optimized backtest on full dataset
        start_date = '2023-01-01'  # Full 2 years of data
        final_balance, trade_count = simple_backtester.run_simple_backtest(
            data, 
            n_steps=100,
            start_date=start_date,
            debug_mode=False  # Full run
        )
        
        if trade_count > 0:
            print(f"\n🎉 SUCCESS! Generated {trade_count} trades")
            
            # Calculate basic performance metrics
            total_return = (final_balance - simple_backtester.initial_capital) / simple_backtester.initial_capital
            closed_trades = [t for t in simple_backtester.trades if t['status'] == 'CLOSED']
            winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
            win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
            
            print(f"\n📊 OPTIMIZED BACKTEST RESULTS")
            print(f"=" * 50)
            print(f"Total Return: {total_return*100:+.2f}%")
            print(f"Final Balance: ${final_balance:,.2f}")
            print(f"Total Trades: {len(closed_trades)}")
            print(f"Win Rate: {win_rate*100:.1f}%")
            
            if closed_trades:
                trade_pnls = [t.get('pnl', 0) for t in closed_trades]
                avg_trade = sum(trade_pnls) / len(trade_pnls)
                print(f"Average Trade: ${avg_trade:.2f}")
                
                if winning_trades and len(winning_trades) < len(closed_trades):
                    losing_trades = [t for t in closed_trades if t.get('pnl', 0) <= 0]
                    gross_profit = sum([t.get('pnl', 0) for t in winning_trades])
                    gross_loss = abs(sum([t.get('pnl', 0) for t in losing_trades]))
                    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                    print(f"Profit Factor: {profit_factor:.2f}")
            
            # Assessment
            if total_return > 0.05:
                print(f"\n✅ STRONG PERFORMANCE - Excellent returns!")
            elif total_return > 0.02:
                print(f"\n👍 GOOD PERFORMANCE - Solid positive returns")
            elif total_return > 0:
                print(f"\n⚠️ MODEST PERFORMANCE - Small positive returns")
            else:
                print(f"\n❌ POOR PERFORMANCE - Negative returns")
                
            print(f"💡 System is working! Ready for live trading consideration.")
        else:
            print(f"\n❌ Still no trades generated")
            print(f"🔍 Need to investigate prediction function further")
            
    except Exception as e:
        print(f"❌ Error during simple backtesting: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 