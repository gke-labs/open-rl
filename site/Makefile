HUGO ?= hugo

.PHONY: preview build deck clean

deck:
	mkdir -p static/deck && cp ../docs/openrl-deck.html static/deck/index.html

preview: deck
	$(HUGO) server -D --disableFastRender

build: deck
	$(HUGO) --minify

clean:
	rm -rf public resources/_gen static/deck .hugo_build.lock
