PACKAGE  ?= ont-currennt
MAJOR    ?= 0
MINOR    ?= 2
SUB      ?= 1
PATCH    ?= 1
CODENAME ?= $(shell lsb_release -cs)
SEDI      = sed -i

deb:
	touch tmp
	rm -rf tmp build *.deb
	mkdir build && cd build && cmake .. && make && strip currennt
	mkdir -p tmp/usr/bin tmp/DEBIAN tmp/usr/share/doc/ont-currennt
	cp deb-src/control.t tmp/DEBIAN/control
	$(SEDI) "s/PACKAGE/$(PACKAGE)/g"   tmp/DEBIAN/control
	$(SEDI) "s/MAJOR/$(MAJOR)/g"       tmp/DEBIAN/control
	$(SEDI) "s/MINOR/$(MINOR)/g"       tmp/DEBIAN/control
	$(SEDI) "s/SUB/$(SUB)/g"           tmp/DEBIAN/control
	$(SEDI) "s/PATCH/$(PATCH)/g"       tmp/DEBIAN/control
	$(SEDI) "s/CODENAME/$(CODENAME)/g" tmp/DEBIAN/control
	cp build/currennt tmp/usr/bin/
	cp README LICENSE NOTICE tmp/usr/share/doc/ont-currennt
	(cd tmp; fakeroot dpkg -b . ../$(PACKAGE)-$(MAJOR).$(MINOR).$(SUB)-$(PATCH)~$(CODENAME).deb)
	rm -rf tmp
