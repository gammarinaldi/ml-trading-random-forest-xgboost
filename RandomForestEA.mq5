//+------------------------------------------------------------------+
//|                                               RandomForestEA.mq5 |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

//--- Include ONNX library
#include <Trade\Trade.mqh>

//--- Input parameters
input double   Lots = 0.1;                // Lot size
input int      StopLoss = 500;             // Stop Loss in points
input int      TakeProfit = 1000;          // Take Profit in points
input int      Magic = 123456;             // Magic number
input string   ModelPath = "Models\\random_forest_model.onnx"; // Path to ONNX model

//--- Global variables
CTrade trade;
long model_handle = INVALID_HANDLE;
datetime last_prediction_time = 0;

//--- Price prediction states
enum ENUM_PRICE_PREDICTION
{
   PRICE_UP = 1,
   PRICE_DOWN = -1,
   PRICE_SAME = 0
};

int ExtPredictedClass = PRICE_SAME;

//--- Model input/output parameters
const long input_shape[] = {1,100};
const long output_shape[] = {1,1};

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   //--- Initialize trade object
   trade.SetExpertMagicNumber(Magic);
   trade.SetMarginMode();
   trade.SetTypeFillingBySymbol(Symbol());
   
   //--- Load ONNX model
   model_handle = OnnxCreateFromFile(ModelPath, ONNX_DEFAULT);
   
   if(model_handle == INVALID_HANDLE)
   {
      Print("Failed to load ONNX model from: ", ModelPath);
      Print("Error: ", GetLastError());
      return INIT_FAILED;
   }
   
   Print("ONNX model loaded successfully from: ", ModelPath);
   
   //--- Set input and output shapes
   if(!OnnxSetInputShape(model_handle, 0, input_shape) || 
      !OnnxSetOutputShape(model_handle, 0, output_shape))
   {
      Print("Failed to set input/output shapes");
      OnnxRelease(model_handle);
      return INIT_FAILED;
   }
   
   Print("Random Forest Expert Advisor initialized successfully");
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   //--- Release ONNX model
   if(model_handle != INVALID_HANDLE)
   {
      OnnxRelease(model_handle);
      model_handle = INVALID_HANDLE;
   }
   
   Print("Random Forest Expert Advisor deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   //--- Check if we have enough bars
   if(Bars(Symbol(), PERIOD_CURRENT) < 100)
      return;
   
   //--- Check if it's time for a new prediction (once per bar)
   datetime current_time = iTime(Symbol(), PERIOD_CURRENT, 0);
   if(current_time == last_prediction_time)
      return;
   
   last_prediction_time = current_time;
   
   //--- Make prediction
   MakePrediction();
   
   //--- Check current positions
   bool has_buy_position = false;
   bool has_sell_position = false;
   
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionSelectByTicket(PositionGetTicket(i)))
      {
         if(PositionGetString(POSITION_SYMBOL) == Symbol() && 
            PositionGetInteger(POSITION_MAGIC) == Magic)
         {
            if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
               has_buy_position = true;
            else if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL)
               has_sell_position = true;
         }
      }
   }
   
   //--- Get current spread
   double spread = SymbolInfoDouble(Symbol(), SYMBOL_SPREAD) * SymbolInfoDouble(Symbol(), SYMBOL_POINT);
   double current_price = SymbolInfoDouble(Symbol(), SYMBOL_BID);
   
   //--- Trading logic based on prediction
   if(ExtPredictedClass == PRICE_UP && !has_buy_position)
   {
      //--- Close any sell positions
      if(has_sell_position)
         CloseAllPositions(POSITION_TYPE_SELL);
      
      //--- Open buy position
      if(spread < 0.000005)
      {
         double sl = StopLoss > 0 ? current_price - StopLoss * SymbolInfoDouble(Symbol(), SYMBOL_POINT) : 0;
         double tp = TakeProfit > 0 ? current_price + TakeProfit * SymbolInfoDouble(Symbol(), SYMBOL_POINT) : 0;
         
         if(trade.Buy(Lots, Symbol(), 0, sl, tp, "RF Prediction: UP"))
         {
            Print("Buy order opened successfully. Prediction: PRICE_UP");
         }
      }
   }
   else if(ExtPredictedClass == PRICE_DOWN && !has_sell_position)
   {
      //--- Close any buy positions
      if(has_buy_position)
         CloseAllPositions(POSITION_TYPE_BUY);
      
      //--- Open sell position
      if(spread < 0.000005)
      {
         double sl = StopLoss > 0 ? current_price + StopLoss * SymbolInfoDouble(Symbol(), SYMBOL_POINT) : 0;
         double tp = TakeProfit > 0 ? current_price - TakeProfit * SymbolInfoDouble(Symbol(), SYMBOL_POINT) : 0;
         
         if(trade.Sell(Lots, Symbol(), 0, sl, tp, "RF Prediction: DOWN"))
         {
            Print("Sell order opened successfully. Prediction: PRICE_DOWN");
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Make prediction using ONNX model                                 |
//+------------------------------------------------------------------+
void MakePrediction()
{
   //--- Prepare input data (last 100 close prices)
   float input_data[];
   ArrayResize(input_data, 100);
   
   for(int i = 0; i < 100; i++)
   {
      input_data[i] = (float)iClose(Symbol(), PERIOD_CURRENT, 99 - i);
   }
   
   //--- Prepare output array
   float output_data[];
   ArrayResize(output_data, 1);
   
   //--- Run prediction
   if(!OnnxRun(model_handle, ONNX_NO_CONVERSION, input_data, output_data))
   {
      Print("ONNX model prediction failed");
      ExtPredictedClass = PRICE_SAME;
      return;
   }
   
   //--- Interpret prediction result
   double spread = SymbolInfoDouble(Symbol(), SYMBOL_SPREAD) * SymbolInfoDouble(Symbol(), SYMBOL_POINT);
   float predicted = output_data[0];
   double current_close = iClose(Symbol(), PERIOD_CURRENT, 1);
   
   if(spread < 0.000005 && predicted > current_close)
   {
      ExtPredictedClass = PRICE_UP;
      Print("Prediction: PRICE_UP (", predicted, " > ", current_close, ")");
   }
   else if(spread < 0.000005 && predicted < current_close)
   {
      ExtPredictedClass = PRICE_DOWN;
      Print("Prediction: PRICE_DOWN (", predicted, " < ", current_close, ")");
   }
   else
   {
      ExtPredictedClass = PRICE_SAME;
      Print("Prediction: PRICE_SAME (", predicted, " ≈ ", current_close, ")");
   }
}

//+------------------------------------------------------------------+
//| Close all positions of specified type                            |
//+------------------------------------------------------------------+
void CloseAllPositions(ENUM_POSITION_TYPE position_type)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionSelectByTicket(PositionGetTicket(i)))
      {
         if(PositionGetString(POSITION_SYMBOL) == Symbol() && 
            PositionGetInteger(POSITION_MAGIC) == Magic &&
            PositionGetInteger(POSITION_TYPE) == position_type)
         {
            trade.PositionClose(PositionGetTicket(i));
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Trade transaction function                                        |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction& trans,
                       const MqlTradeRequest& request,
                       const MqlTradeResult& result)
{
   //--- Handle trade transactions if needed
   if(trans.symbol == Symbol() && trans.magic == Magic)
   {
      if(trans.type == TRADE_TRANSACTION_DEAL_ADD)
      {
         Print("Trade executed: ", EnumToString((ENUM_DEAL_TYPE)trans.deal_type), 
               " Volume: ", trans.volume, " Price: ", trans.price);
      }
   }
} 