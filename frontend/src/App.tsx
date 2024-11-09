// frontend/src/App.tsx
import React, { useState } from 'react';
import { 
  Stack, 
  TextField, 
  PrimaryButton, 
  Text, 
  MessageBar, 
  MessageBarType,
  Spinner,
  IStackTokens
} from '@fluentui/react';
import axios from 'axios';

interface ChatResponse {
  status: string;
  response: string;
  context?: Array<{
    text: string;
    metadata: {
      source: string;
      chunk_index: number;
    };
  }>;
}

const stackTokens: IStackTokens = { 
  childrenGap: 10,
  padding: 20
};

const App: React.FC = () => {
  const [message, setMessage] = useState('');
  const [response, setResponse] = useState('');
  const [context, setContext] = useState<ChatResponse['context']>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResponse('');
    setContext([]);
    
    try {
      const { data } = await axios.post<ChatResponse>('/api/chat', { message });
      if (data.status === 'success') {
        setResponse(data.response);
        setContext(data.context || []);
      } else {
        throw new Error(data.response);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error processing your request');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Stack tokens={stackTokens} style={{ maxWidth: 800, margin: '0 auto' }}>
      <Text variant="xxLarge">RAG Chatbot</Text>
      
      {error && (
        <MessageBar messageBarType={MessageBarType.error} onDismiss={() => setError('')}>
          {error}
        </MessageBar>
      )}

      <form onSubmit={handleSubmit}>
        <Stack tokens={{ childrenGap: 10 }}>
          <TextField
            value={message}
            onChange={(_, newValue) => setMessage(newValue || '')}
            placeholder="Ask a question..."
            multiline
            autoAdjustHeight
            disabled={loading}
          />
          <PrimaryButton 
            type="submit" 
            disabled={loading || !message.trim()}
          >
            {loading ? <Spinner label="Processing..." /> : 'Send'}
          </PrimaryButton>
        </Stack>
      </form>

      {response && (
        <Stack tokens={{ childrenGap: 10 }}>
          <Text variant="large">Response:</Text>
          <Text>{response}</Text>
          
          {context && context.length > 0 && (
            <>
              <Text variant="large">Source Documents:</Text>
              {context.map((doc, i) => (
                <Stack key={i} style={{ backgroundColor: '#f8f8f8', padding: 10 }}>
                  <Text variant="medium">Source: {doc.metadata.source}</Text>
                  <Text>{doc.text}</Text>
                </Stack>
              ))}
            </>
          )}
        </Stack>
      )}
    </Stack>
  );
};

export default App;