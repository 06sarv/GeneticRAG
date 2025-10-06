import { useState } from 'react';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');

  const handleSendMessage = async () => {
    if (input.trim()) {
      const newMessage = { text: input, sender: 'user' };
      setMessages((prevMessages) => [...prevMessages, newMessage]);
      setInput('');

      try {
        const response = await fetch('http://localhost:8000/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ message: input }),
        });
        const data = await response.json();
        setMessages((prevMessages) => [...prevMessages, { text: data.reply, sender: 'bot' }]);
      } catch (error) {
        console.error('Error sending message:', error);
        setMessages((prevMessages) => [...prevMessages, { text: 'Error: Could not connect to the server.', sender: 'bot' }]);
      }
    }
  };

  return (
    <div style={{
      fontFamily: 'Arial, sans-serif',
      display: 'flex',
      flexDirection: 'column',
      height: '100vh',
      margin: 0,
      backgroundColor: '#f0f2f5',
    }}>
      <header style={{
        backgroundColor: '#007bff',
        color: 'white',
        padding: '15px 20px',
        textAlign: 'center',
        fontSize: '24px',
        fontWeight: 'bold',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
      }}>
        Genetic Counseling Chatbot
      </header>
      <div style={{
        flexGrow: 1,
        padding: '20px',
        overflowY: 'auto',
        display: 'flex',
        flexDirection: 'column',
        gap: '10px',
      }}>
        {messages.map((msg, index) => (
          <div
            key={index}
            style={{
              alignSelf: msg.sender === 'user' ? 'flex-end' : 'flex-start',
              backgroundColor: msg.sender === 'user' ? '#dcf8c6' : '#ffffff',
              padding: '10px 15px',
              borderRadius: '18px',
              maxWidth: '70%',
              boxShadow: '0 1px 2px rgba(0,0,0,0.1)',
              wordBreak: 'break-word',
            }}
          >
            {msg.text}
          </div>
        ))}
      </div>
      <div style={{
        display: 'flex',
        padding: '15px 20px',
        backgroundColor: 'white',
        borderTop: '1px solid #e0e0e0',
        boxShadow: '0 -2px 4px rgba(0,0,0,0.05)',
      }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
          placeholder="Type your message..."
          style={{
            flexGrow: 1,
            padding: '12px 15px',
            border: '1px solid #ced4da',
            borderRadius: '20px',
            marginRight: '10px',
            fontSize: '16px',
            outline: 'none',
          }}
        />
        <button
          onClick={handleSendMessage}
          style={{
            backgroundColor: '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '20px',
            padding: '12px 20px',
            fontSize: '16px',
            cursor: 'pointer',
            transition: 'background-color 0.2s ease-in-out',
          }}
        >
          Send
        </button>
      </div>
    </div>
  );
}

export default App;
