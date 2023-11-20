# add poetry to the zsh plugins
cat ~/.zshrc | sed -e "s/plugins=(git)/plugins=(git poetry)/" > ~/temp
cat ~/temp > ~/.zshrc

poetry install
