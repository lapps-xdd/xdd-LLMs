# To start Ollama I would like to use "systemctl start ollama", but this gives
# an error on the container so using a somewhat crappy alternative.

echo "Starting Ollama"
ollama start &
sleep 5

echo "Pulling model llama3"
ollama pull llama3
ollama list
