PACKAGE  ?= ont-currennt
MAJOR    ?= 0
MINOR    ?= 2
PATCH    ?= 1
CODENAME ?= $(shell lsb_release -cs)
SEDI      = sed -i

deb:
	mkdir build && cd build && cmake .. && make && strip currennt
	touch tmp
	rm -rf tmp
	mkdir -p tmp/usr/bin tmp/DEBIAN
	cp deb-src/control.t tmp/DEBIAN/control
	$(SEDI) "s/PACKAGE/$(PACKAGE)/g" tmp/DEBIAN/control
	$(SEDI) "s/MAJOR/$(MAJOR)/g" tmp/DEBIAN/control
	$(SEDI) "s/MINOR/$(MINOR)/g" tmp/DEBIAN/control
	$(SEDI) "s/PATCH/$(PATCH)/g" tmp/DEBIAN/control
	$(SEDI) "s/CODENAME/$(CODENAME)/g" tmp/DEBIAN/control
	cp build/currennt tmp/usr/bin/
	(cd tmp; fakeroot dpkg -b . ../$(PACKAGE)-$(MAJOR).$(MINOR)-$(PATCH)~$(CODENAME).deb)
	rm -rf tmp
