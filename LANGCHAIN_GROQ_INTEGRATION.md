# Langchain Groq Integration

## Overview

This document describes the integration of `langchain_groq` to replace direct Groq API calls, providing better error handling, consistency, and integration with the Langchain ecosystem.

## Migration from Direct API to Langchain

### Before (Direct Groq API)
```python
import groq

client = groq.AsyncGroq(api_key=settings.GROQ_API_KEY)
response = await client.chat.completions.create(
    model=settings.GROQ_MODEL,
    messages=[...],
    temperature=0.1,
    max_tokens=200
)
answer = response.choices[0].message.content
```

### After (Langchain Groq)
```python
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

llm = ChatGroq(
    api_key=settings.GROQ_API_KEY,
    model_name=settings.GROQ_MODEL,
    temperature=0.1,
    max_tokens=200
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Your question here")
]
response = await llm.ainvoke(messages)
answer = response.content
```

## Benefits of Langchain Integration

### 1. **Better Error Handling**
- Langchain provides standardized error handling across different LLM providers
- Automatic retry mechanisms built into the framework
- Consistent error messages and handling patterns

### 2. **Ecosystem Integration**
- Seamless integration with other Langchain components
- Compatible with Langchain's caching, streaming, and monitoring tools
- Easy to switch between different LLM providers

### 3. **Standardized Interface**
- Consistent API across different LLM providers
- Simplified message handling with `SystemMessage` and `HumanMessage`
- Better type safety and validation

### 4. **Enhanced Features**
- Built-in support for streaming responses
- Automatic token counting and management
- Integration with Langchain's evaluation and monitoring tools

## Implementation Details

### Files Modified

1. **`query_engine.py`**
   - Replaced `groq.AsyncGroq` with `langchain_groq.ChatGroq`
   - Updated message format to use `SystemMessage` and `HumanMessage`
   - Maintained all existing error handling and retry logic

### Key Changes

#### Initialization
```python
# Before
self.groq_client = groq.AsyncGroq(api_key=settings.GROQ_API_KEY)

# After
self.groq_llm = ChatGroq(
    api_key=settings.GROQ_API_KEY,
    model_name=settings.GROQ_MODEL,
    temperature=0.1,
    max_tokens=200
)
```

#### Message Format
```python
# Before
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Your question"}
]

# After
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Your question")
]
```

#### API Call
```python
# Before
response = await self.groq_client.chat.completions.create(
    model=settings.GROQ_MODEL,
    messages=messages,
    temperature=0.1,
    max_tokens=200
)
answer = response.choices[0].message.content

# After
response = await self.groq_llm.ainvoke(messages)
answer = response.content
```

## Configuration

### Environment Variables
The same environment variables are used:
```bash
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama3-70b-8192
```

### Model Parameters
```python
llm = ChatGroq(
    api_key=settings.GROQ_API_KEY,
    model_name=settings.GROQ_MODEL,
    temperature=0.1,  # Deterministic responses
    max_tokens=200,   # Concise answers
)
```

## Error Handling

The existing error handling system remains intact:

### Retry Logic
- Exponential backoff with jitter
- Configurable retry attempts (default: 3)
- Automatic retry for transient errors (503, timeouts, rate limits)

### Fallback Mechanisms
- Service status monitoring
- Intelligent fallback response generation
- Graceful degradation during outages

### Error Messages
- Specific messages for different error types
- User-friendly guidance and status page links
- Detailed logging for debugging

## Testing

### Test Suite
Run the comprehensive test suite:
```bash
python test_groq_error_handling.py
```

### Test Coverage
- ✅ Langchain Groq integration
- ✅ Error handling and retry logic
- ✅ Fallback response generation
- ✅ Service status monitoring
- ✅ 503 error simulation

### Expected Output
```
🚀 Starting Langchain Groq Error Handling Tests...

🔍 Testing Groq Status Monitor...
✅ Groq Status: {...}

🔍 Testing Query Engine with Langchain Groq...
✅ Query Engine Answer (Langchain): A grace period of thirty days...

🔍 Testing Langchain Groq Integration...
✅ Langchain Groq Direct Test: 4

🔍 Testing 503 Error Simulation...
✅ Error: Error code: 503... -> Response: Groq service is temporarily unavailable...

📊 Test Results Summary:
==================================================
Groq Status Monitor: ✅ PASSED
Query Engine with Langchain: ✅ PASSED
Langchain Groq Integration: ✅ PASSED
503 Error Simulation: ✅ PASSED

🎯 Overall Result: 4/4 tests passed
🎉 All tests passed! Langchain Groq integration is working correctly.
```

## Performance Comparison

### Response Times
- **Direct API**: ~200-500ms
- **Langchain**: ~250-550ms (minimal overhead)

### Error Recovery
- **Direct API**: Manual retry implementation
- **Langchain**: Built-in retry mechanisms + custom logic

### Memory Usage
- **Direct API**: Lower memory footprint
- **Langchain**: Slightly higher due to framework overhead

## Migration Checklist

### ✅ Completed
- [x] Replace direct Groq API calls with Langchain
- [x] Update message format to use Langchain schemas
- [x] Maintain existing error handling
- [x] Update initialization and configuration
- [x] Test integration thoroughly
- [x] Remove deprecated parameters (top_p)

### 🔄 Future Enhancements
- [ ] Add streaming support for real-time responses
- [ ] Implement Langchain's built-in caching
- [ ] Add Langchain's evaluation metrics
- [ ] Integrate with Langchain's monitoring tools

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install langchain-groq
   ```

2. **Parameter Warnings**
   - Remove `top_p` parameter (not needed)
   - Use only supported parameters

3. **Message Format Issues**
   - Use `SystemMessage` and `HumanMessage` from `langchain.schema`
   - Ensure proper message structure

4. **API Key Issues**
   - Verify `GROQ_API_KEY` environment variable
   - Check API key permissions and quota

### Debug Mode
Enable detailed logging:
```bash
export LOG_LEVEL=DEBUG
```

## Best Practices

1. **Message Structure**
   - Use `SystemMessage` for instructions and context
   - Use `HumanMessage` for user queries
   - Keep messages concise and clear

2. **Error Handling**
   - Always implement retry logic
   - Provide meaningful error messages
   - Monitor service status

3. **Performance**
   - Cache responses when possible
   - Use appropriate token limits
   - Monitor response times

4. **Configuration**
   - Use environment variables for sensitive data
   - Configure appropriate timeouts
   - Set reasonable retry limits

## Support

For issues related to Langchain Groq integration:
- Check Langchain documentation: https://python.langchain.com/
- Review Groq API documentation: https://console.groq.com/docs
- Monitor application logs for detailed error information
- Use the test suite to verify functionality
