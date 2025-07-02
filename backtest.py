import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use('Agg')

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Additional imports for parameter optimization
import itertools
import random

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
        
        # Dynamic risk management parameters
        self.atr_period = 14           # ATR calculation period
        self.atr_stop_multiplier = 1.5 # Stop loss = ATR * multiplier
        self.atr_profit_multiplier = 3.0 # Take profit = ATR * multiplier (2:1 reward:risk)
        
        self.trades = []
        self.equity_curve = []
        
    def load_model(self, model_path):
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"✅ Model loaded successfully")
            return model
        else:
            raise FileNotFoundError(f"❌ Model not found at {model_path}")
    
    def calculate_atr(self, df, period=14):
        """
        Calculate Average True Range (ATR)
        ATR = moving average of True Range over specified period
        """
        # For hourly data, we need high, low, close columns
        # Since we only have close prices, we'll simulate high/low based on volatility
        if 'high' not in df.columns or 'low' not in df.columns:
            # Estimate high/low from close price and volatility
            volatility = df['close'].rolling(window=5).std()
            df['high'] = df['close'] + volatility * 0.5
            df['low'] = df['close'] - volatility * 0.5
        
        # Calculate True Range
        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate ATR as exponential moving average of True Range
        df['atr'] = df['true_range'].ewm(span=period, adjust=False).mean()
        
        # Clean up temporary columns
        df.drop(['prev_close', 'tr1', 'tr2', 'tr3', 'true_range'], axis=1, inplace=True)
        
        return df
    

    
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
        
        # Calculate ATR for dynamic risk management
        df = self.calculate_atr(df, self.atr_period)
        
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
    
    def calculate_position_size(self, confidence, account_balance, atr_value):
        """
        Calculate position size based on ATR-based stop loss
        """
        risk_amount = account_balance * self.max_position_size
        stop_loss_distance = atr_value * self.atr_stop_multiplier
        
        # Calculate lot size based on risk amount and stop loss distance
        # For EURUSD, 1 pip = 0.0001, and 1 standard lot = 100,000 units
        # Position value per pip = lot_size * 100,000 * 0.0001 = lot_size * 10
        pip_value_per_lot = 10  # For EURUSD
        stop_loss_pips = stop_loss_distance / 0.0001  # Convert to pips
        
        if stop_loss_pips > 0:
            lot_size = risk_amount / (stop_loss_pips * pip_value_per_lot)
            lot_size = max(0.01, min(lot_size, 0.5))  # Min 0.01, max 0.5 lots
        else:
            lot_size = 0.01  # Default minimum
        
        return lot_size
    
    def execute_trade(self, signal, predicted_price, confidence, current_price, account_balance, timestamp, recent_data):
        """
        Execute trade with dynamic ATR-based stop loss and take profit
        """
        # Get current ATR value
        current_row = recent_data.iloc[-1]
        atr_value = current_row.get('atr', 0.0001)  # Default fallback
        
        # Calculate dynamic stop loss and take profit distances using ATR
        stop_loss_distance = atr_value * self.atr_stop_multiplier
        take_profit_distance = atr_value * self.atr_profit_multiplier
        
        # Calculate position size based on ATR
        lot_size = self.calculate_position_size(confidence, account_balance, atr_value)
        
        pip_value = 0.0001
        if signal == 1:  # BUY
            entry_price = current_price + (self.spread_pips / 2) * pip_value
            stop_loss = entry_price - stop_loss_distance
            take_profit = entry_price + take_profit_distance
        else:  # SELL
            entry_price = current_price - (self.spread_pips / 2) * pip_value
            stop_loss = entry_price + stop_loss_distance
            take_profit = entry_price - take_profit_distance
        
        transaction_cost = (self.spread_pips + self.slippage_pips) * pip_value * lot_size * 100000
        
        # Calculate risk-reward ratio for information
        risk_distance = abs(entry_price - stop_loss)
        reward_distance = abs(take_profit - entry_price)
        risk_reward_ratio = reward_distance / risk_distance if risk_distance > 0 else 0
        
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
            'atr_value': atr_value,
            'stop_loss_pips': stop_loss_distance / pip_value,
            'take_profit_pips': take_profit_distance / pip_value,
            'risk_reward_ratio': risk_reward_ratio,
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
    
    def run_backtest(self, data, n_steps=100, start_date=None, debug_mode=True):
        """Run backtest with debugging"""
        print("🚀 Starting backtest with minimal filtering...")
        
        data = data.copy()
        data = self.add_technical_indicators(data, 'close')
        data = data.dropna()
        
        if start_date:
            data = data[data.index >= start_date]
        
        print(f"📊 Backtesting period: {data.index[0]} to {data.index[-1]}")
        print(f"📈 Total periods: {len(data)}")
        
        # Exclude ATR from model features - it's only for risk management
        feature_cols = [col for col in data.columns if col not in ['close', 'atr', 'high', 'low']]
        print(f"📊 Features for ML model: {len(feature_cols)} (excluding ATR risk management indicator)")
        
        account_balance = self.initial_capital
        self.trades = []
        self.equity_curve = []
        
        total_signals = 0
        valid_predictions = 0
        confidence_above_threshold = 0
        
        # Test periods - full dataset when not debugging
        test_periods = min(1000, len(data) - n_steps) if debug_mode else len(data) - n_steps
        
        print(f"🔍 Testing {test_periods} periods...")
        
        with tqdm(total=test_periods, desc="Backtesting", unit="periods") as pbar:
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
                                    account_balance, current_timestamp, recent_data
                                )
                                
                                if new_trade:
                                    self.trades.append(new_trade)
                                    account_balance -= new_trade['transaction_cost']
                                    
                                    if debug_mode and len(self.trades) <= 5:
                                        print(f"  📈 Trade {len(self.trades)}: {new_trade['signal']} at {current_price:.5f}")
                                        print(f"     📊 ATR: {new_trade['atr_value']:.5f}")
                                        print(f"     🛡️ Stop: {new_trade['stop_loss_pips']:.1f} pips, 🎯 Target: {new_trade['take_profit_pips']:.1f} pips")
                                        print(f"     ⚖️ Risk:Reward = 1:{new_trade['risk_reward_ratio']:.2f}, Confidence: {confidence:.6f}")
                
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
        
        print(f"✅ Backtest completed!")
        print(f"📊 Total signals: {total_signals}")
        print(f"📊 Valid predictions: {valid_predictions}")
        print(f"📊 Confidence above threshold: {confidence_above_threshold}")
        print(f"📈 Executed trades: {len(closed_trades)}")
        print(f"💰 Final balance: ${account_balance:,.2f}")
        print(f"📊 Total return: {((account_balance - self.initial_capital) / self.initial_capital * 100):+.2f}%")
        
        return account_balance, len(closed_trades)

class ParameterOptimizer:
    """
    Parameter optimization system for finding optimal trading parameters
    """
    
    def __init__(self, data, model_path='models/unified_price_regressor.pkl', initial_capital=10000):
        self.data = data
        self.model_path = model_path
        self.initial_capital = initial_capital
        self.results = []
        
    def define_parameter_space(self):
        """Define the parameter ranges to optimize"""
        return {
            'atr_period': [10, 14, 20, 25],  # ATR calculation periods
            'atr_stop_multiplier': [1.0, 1.5, 2.0, 2.5],  # Stop loss multipliers
            'atr_profit_multiplier': [2.0, 2.5, 3.0, 3.5, 4.0],  # Take profit multipliers
            'min_confidence': [0.0003, 0.0005, 0.0008, 0.001, 0.0015],  # Confidence thresholds
            'max_position_size': [0.003, 0.005, 0.008, 0.01]  # Position size percentages
        }
    
    def calculate_objective_score(self, final_balance, trades, initial_capital):
        """
        Calculate optimization objective score
        Combines multiple metrics for robust optimization
        """
        if len(trades) == 0:
            return -999999  # Heavily penalize no trades
        
        # Basic metrics
        total_return = (final_balance - initial_capital) / initial_capital
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        # Calculate profit factor
        gross_profit = sum([t.get('pnl', 0) for t in winning_trades])
        losing_trades = [t for t in trades if t.get('pnl', 0) <= 0]
        gross_loss = abs(sum([t.get('pnl', 0) for t in losing_trades]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average trade return
        avg_trade_return = sum([t.get('pnl', 0) for t in trades]) / len(trades)
        
        # Risk-adjusted score combining multiple factors
        base_score = total_return * 100  # Base return percentage
        
        # Bonuses and penalties
        win_rate_bonus = (win_rate - 0.3) * 50 if win_rate > 0.3 else (win_rate - 0.3) * 100  # Bonus for good win rate
        profit_factor_bonus = (profit_factor - 1.0) * 20 if profit_factor > 1.0 else (profit_factor - 1.0) * 50  # Bonus for profit factor > 1
        trade_count_bonus = min(len(trades) / 100, 3)  # Small bonus for reasonable trade count
        
        # Penalty for too few trades
        if len(trades) < 20:
            trade_penalty = (20 - len(trades)) * -2
        else:
            trade_penalty = 0
        
        objective_score = base_score + win_rate_bonus + profit_factor_bonus + trade_count_bonus + trade_penalty
        
        return objective_score
    
    def run_single_optimization(self, params, test_subset=True):
        """Run backtest with specific parameter set"""
        try:
            # Create backtester with custom parameters
            backtester = Backtester(self.model_path, self.initial_capital)
            backtester.atr_period = params['atr_period']
            backtester.atr_stop_multiplier = params['atr_stop_multiplier']
            backtester.atr_profit_multiplier = params['atr_profit_multiplier']
            backtester.min_confidence = params['min_confidence']
            backtester.max_position_size = params['max_position_size']
            
            # Run backtest on subset for optimization (faster)
            test_data = self.data.copy()
            if test_subset:
                # Use last 6 months for optimization testing
                start_date = '2023-07-01'
                test_data = test_data[test_data.index >= start_date]
            
            final_balance, trade_count = backtester.run_backtest(
                test_data, 
                n_steps=100,
                start_date=None,  # Already filtered above
                debug_mode=False
            )
            
            closed_trades = [t for t in backtester.trades if t['status'] == 'CLOSED']
            objective_score = self.calculate_objective_score(final_balance, closed_trades, self.initial_capital)
            
            return {
                'params': params.copy(),
                'final_balance': final_balance,
                'total_return': (final_balance - self.initial_capital) / self.initial_capital,
                'trade_count': trade_count,
                'objective_score': objective_score,
                'win_rate': len([t for t in closed_trades if t.get('pnl', 0) > 0]) / len(closed_trades) if closed_trades else 0,
                'trades': closed_trades
            }
            
        except Exception as e:
            print(f"  ❌ Error with params {params}: {e}")
            return {
                'params': params.copy(),
                'final_balance': self.initial_capital,
                'total_return': 0,
                'trade_count': 0,
                'objective_score': -999999,
                'win_rate': 0,
                'trades': []
            }
    
    def optimize_parameters(self, max_combinations=200):
        """
        Run parameter optimization using grid search with sampling
        """
        print("🔧 Starting parameter optimization...")
        
        param_space = self.define_parameter_space()
        
        # Generate all combinations
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        all_combinations = list(itertools.product(*param_values))
        
        print(f"📊 Total possible combinations: {len(all_combinations)}")
        
        # Sample combinations if too many
        if len(all_combinations) > max_combinations:
            random.seed(42)  # For reproducibility
            combinations_to_test = random.sample(all_combinations, max_combinations)
            print(f"🎲 Randomly sampling {max_combinations} combinations for optimization")
        else:
            combinations_to_test = all_combinations
            print(f"🔍 Testing all {len(combinations_to_test)} combinations")
        
        self.results = []
        
        print(f"⏳ Running optimization tests...")
        with tqdm(total=len(combinations_to_test), desc="Parameter Optimization", unit="tests") as pbar:
            for i, combination in enumerate(combinations_to_test):
                params = dict(zip(param_names, combination))
                
                result = self.run_single_optimization(params, test_subset=True)
                self.results.append(result)
                
                # Update progress bar with best score so far
                if self.results:
                    best_score = max([r['objective_score'] for r in self.results])
                    pbar.set_postfix({'Best Score': f'{best_score:.2f}'})
                
                pbar.update(1)
        
        # Sort by objective score
        self.results.sort(key=lambda x: x['objective_score'], reverse=True)
        
        print(f"✅ Parameter optimization completed!")
        return self.get_best_parameters()
    
    def get_best_parameters(self, top_n=5):
        """Get the best parameter sets"""
        if not self.results:
            return None
        
        print(f"\n🏆 TOP {top_n} PARAMETER SETS:")
        print("=" * 100)
        
        best_results = self.results[:top_n]
        
        for i, result in enumerate(best_results, 1):
            params = result['params']
            print(f"\n#{i} - Score: {result['objective_score']:.2f}")
            print(f"   📊 Return: {result['total_return']*100:+.2f}% | Trades: {result['trade_count']} | Win Rate: {result['win_rate']*100:.1f}%")
            print(f"   ⚙️ ATR Period: {params['atr_period']} | Stop Mult: {params['atr_stop_multiplier']} | Profit Mult: {params['atr_profit_multiplier']}")
            print(f"   💰 Min Confidence: {params['min_confidence']:.4f} | Position Size: {params['max_position_size']*100:.1f}%")
        
        return best_results[0]['params']  # Return best parameters
    
    def validate_best_parameters(self, best_params):
        """
        Validate best parameters on full dataset
        """
        print(f"\n🧪 VALIDATING BEST PARAMETERS ON FULL DATASET...")
        print("=" * 60)
        
        # Run full backtest with best parameters
        backtester = Backtester(self.model_path, self.initial_capital)
        backtester.atr_period = best_params['atr_period']
        backtester.atr_stop_multiplier = best_params['atr_stop_multiplier']
        backtester.atr_profit_multiplier = best_params['atr_profit_multiplier']
        backtester.min_confidence = best_params['min_confidence']
        backtester.max_position_size = best_params['max_position_size']
        
        print(f"🔧 OPTIMIZED PARAMETERS:")
        print(f"  📊 ATR Period: {best_params['atr_period']} periods")
        print(f"  🛡️ Stop Loss: ATR × {best_params['atr_stop_multiplier']}")
        print(f"  🎯 Take Profit: ATR × {best_params['atr_profit_multiplier']}")
        print(f"  💰 Min Confidence: {best_params['min_confidence']:.4f}")
        print(f"  📏 Position Size: {best_params['max_position_size']*100:.1f}%")
        
        # Run full validation backtest
        start_date = '2023-01-01'
        final_balance, trade_count = backtester.run_backtest(
            self.data, 
            n_steps=100,
            start_date=start_date,
            debug_mode=False
        )
        
        if trade_count > 0:
            closed_trades = [t for t in backtester.trades if t['status'] == 'CLOSED']
            winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
            
            total_return = (final_balance - self.initial_capital) / self.initial_capital
            win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
            
            print(f"\n🎉 OPTIMIZED SYSTEM VALIDATION RESULTS:")
            print(f"=" * 60)
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
                
                # ATR statistics
                atr_values = [t.get('atr_value', 0) for t in closed_trades if t.get('atr_value')]
                if atr_values:
                    stop_loss_pips = [t.get('stop_loss_pips', 0) for t in closed_trades if t.get('stop_loss_pips')]
                    take_profit_pips = [t.get('take_profit_pips', 0) for t in closed_trades if t.get('take_profit_pips')]
                    risk_reward_ratios = [t.get('risk_reward_ratio', 0) for t in closed_trades if t.get('risk_reward_ratio')]
                    
                    print(f"\n📊 OPTIMIZED RISK MANAGEMENT STATS:")
                    print(f"Average ATR: {sum(atr_values)/len(atr_values):.5f}")
                    print(f"Average Stop Loss: {sum(stop_loss_pips)/len(stop_loss_pips):.1f} pips")
                    print(f"Average Take Profit: {sum(take_profit_pips)/len(take_profit_pips):.1f} pips")
                    print(f"Average Risk:Reward: 1:{sum(risk_reward_ratios)/len(risk_reward_ratios):.2f}")
            
            # Performance assessment
            if total_return > 0.1:
                print(f"\n🚀 EXCELLENT! Optimized system shows strong performance!")
            elif total_return > 0.05:
                print(f"\n✅ GOOD! Optimized system shows solid positive returns!")
            elif total_return > 0:
                print(f"\n👍 POSITIVE! Optimized system shows modest gains!")
            else:
                print(f"\n⚠️ Optimization helped but system still needs work")
            
            print(f"💡 Parameter optimization complete! System ready for further refinement.")
            
        return final_balance, trade_count

def main():
    """Run the backtesting system with optional parameter optimization"""
    print("🚀 Advanced Forex Trading System with Parameter Optimization\n")
    
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
        
        # Ask user for optimization preference
        print(f"\n🔧 SYSTEM OPTIONS:")
        print(f"1️⃣ Run with default parameters (faster)")
        print(f"2️⃣ Run parameter optimization first (recommended, slower)")
        print(f"3️⃣ Run both optimization and validation")
        
        # For automation, default to option 3 (full optimization)
        choice = "3"  # You can change this to "1" for quick testing or make it interactive
        
        if choice == "1":
            # Run with default parameters only
            print(f"⚡ Running with default parameters...")
            
            backtester = Backtester(
                model_path='models/unified_price_regressor.pkl',
                initial_capital=10000
            )
            
            print(f"\n🔧 DEFAULT PARAMETERS:")
            print(f"  📊 ATR Period: {backtester.atr_period} periods")
            print(f"  🛡️ Stop Loss: ATR × {backtester.atr_stop_multiplier}")
            print(f"  🎯 Take Profit: ATR × {backtester.atr_profit_multiplier}")
            print(f"  💰 Min Confidence: {backtester.min_confidence:.4f}")
            print(f"  📏 Position Size: {backtester.max_position_size*100:.1f}%")
            
            start_date = '2023-01-01'
            final_balance, trade_count = backtester.run_backtest(
                data, 
                n_steps=100,
                start_date=start_date,
                debug_mode=False
            )
            
            # Show results
            if trade_count > 0:
                print(f"\n🎉 Generated {trade_count} trades with default parameters")
                total_return = (final_balance - backtester.initial_capital) / backtester.initial_capital
                print(f"📊 Total Return: {total_return*100:+.2f}%")
                print(f"💰 Final Balance: ${final_balance:,.2f}")
            
        elif choice in ["2", "3"]:
            # Run parameter optimization
            print(f"🔧 Running parameter optimization...")
            
            optimizer = ParameterOptimizer(
                data=data,
                model_path='models/unified_price_regressor.pkl',
                initial_capital=10000
            )
            
            # Run optimization (testing 200 random combinations for speed)
            best_params = optimizer.optimize_parameters(max_combinations=200)
            
            if choice == "3" and best_params:
                # Validate best parameters on full dataset
                optimizer.validate_best_parameters(best_params)
            elif best_params:
                print(f"\n✅ Parameter optimization completed!")
                print(f"💡 Best parameters found. Use choice=3 to run full validation.")
        
        else:
            print(f"❌ Invalid choice. Please run again and select 1, 2, or 3.")
            
    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 