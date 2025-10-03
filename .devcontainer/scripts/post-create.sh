sudo apt-get update
sudo apt-get install -y tig procps
sudo apt-get clean
rm -rf /var/lib/apt/lists/*

pip install --no-cache-dir uv==0.8.22

# claude-code, gemini-cli, lefthook をインストール
npm install -g @anthropic-ai/claude-code@2.0.5 @google/gemini-cli@0.6.1 lefthook@1.13.4

# CCPluginsをインストール
curl -sSL https://raw.githubusercontent.com/brennercruvinel/CCPlugins/main/install.sh | bash

lefthook install
