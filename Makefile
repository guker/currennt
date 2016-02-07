#!/usr/bin/make -f
SHELL = /bin/sh
MAJOR    := 1
MINOR    := 0
PATCH    ?= 0
CODENAME ?= $(shell lsb_release -cs)
SEDI      = sed -i
BUILD     = build
CURRENNT  = $(BUILD)/currennt

deb:
	mkdir -p tmp/usr/local/bin
	mkdir -p tmp/DEBIAN
	cp deb-src/control.tmpl tmp/DEBIAN/control
	$(SEDI) "s/MAJOR/$(MAJOR)/g" tmp/DEBIAN/control
	$(SEDI) "s/MINOR/$(MINOR)/g" tmp/DEBIAN/control
	$(SEDI) "s/PATCH/$(PATCH)/g" tmp/DEBIAN/control
	$(SEDI) "s/CODENAME/$(CODENAME)/g" tmp/DEBIAN/control
	cp $(BUILD)/currennt tmp/usr/local/bin
	(cd tmp; fakeroot dpkg -b . ../currennt-$(MAJOR).$(MINOR)-$(PATCH)~$(CODENAME).deb)

$(CURRENNT):
	(mkdir build; cd build; cmake ..; make;)

name:
	echo "$(MAJOR).$(MINOR)-$(PATCH)~$(CODENAME)"

clean:
	rm -rf build tmp

.PHONY: clean deb name
