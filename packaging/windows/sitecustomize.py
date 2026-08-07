"""F221 — the shipped interpreter trusts the certificate set that ships with it.

Copied into the payload at `python\\Lib\\site-packages\\sitecustomize.py`, beside the
`_sorta_lib.pth` that finds `lib\\`. It is not part of the Python package: a
`uv tool install` of Sorta uses a Python somebody else set up, and this file is about the
interpreter WE ship.

What it is for
--------------
Windows verifies a TLS chain against the SYSTEM root certificate store, and on a freshly
installed Windows that store is nearly empty — Windows fetches roots on demand, and on a
clean machine that regularly does not happen. So on the owner's clean virtual machine the
first thing the product does, download the weights a tier needs, died with

    <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
     unable to get local issuer certificate (_ssl.c:1032)>

and with it every download there is: the faces tier, the search tier, the deep tier, the
CLIP weights. The whole tiered construction was unreachable on the machine it was built
for.

`certifi` already travels inside the payload (`lib\\certifi\\cacert.pem`) because
`requests` and `huggingface_hub` depend on it and use it by default. Plain
`urllib.request`, which is what the weight download goes through, has never heard of it
and asks Windows instead. Pointing OpenSSL at that same file is what makes all three
download paths trust the same thing rather than two of them trusting one thing and the
third trusting the state of the machine.

Why a `sitecustomize` and not a launcher, a shortcut or an entry point
---------------------------------------------------------------------
There are five ways this program starts: the two Start-menu shortcuts, `sorta-setup`, the
console command and the tray. A variable set in one of them fixes one route out of five.
CPython imports `sitecustomize` from `site-packages` while it is still starting up,
before a line of our code runs, on every one of the five — it is the one place they all
pass through. It also changes nothing outside this process: `os.environ` here is this
interpreter's own environment, nothing is written to the machine, and no other program's
environment is touched.

Why `setdefault` and not assignment
-----------------------------------
A corporate proxy with a root of its own is an ordinary thing, and somebody who has
already named their own certificate set must keep it.

Why the path is derived from `__file__`
---------------------------------------
The payload is built on one machine and copied to another — the same reason `lib\\` is
found through a relative `.pth` beside this file. An absolute path written in at build
time would name a directory that exists only on the machine that built it.

What this deliberately does NOT do
----------------------------------
There is no branch here that switches verification off, and there must never be one: a
program that downloads and then RUNS model weights has to know where they came from.
`truststore` was the other candidate and is the wrong one for exactly the reason above —
it defers to the operating system's store, and the operating system's store is what is
empty on a clean machine. certifi's set is self-contained and versioned with the
delivery, so it does not depend on the state of the machine at all, which is the same
principle the missing MSVC runtime was fixed with (F218).
"""
import os

# ...\\python\\Lib\\site-packages\\sitecustomize.py -> the installation directory. Four
# levels, and the `.pth` beside this file climbs the same four.
_INSTALL = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
_CERTIFI = os.path.join(_INSTALL, "lib", "certifi")
_BUNDLE = os.path.join(_CERTIFI, "cacert.pem")

# Silent when the file is not there rather than loud: this runs at the start of EVERY
# process, including the tray, which has no console to be loud into. That the set is
# actually in the payload is the build's job — `payload_trust_gap` in
# `scripts/build_installer.py` refuses to compile an installer without it.
if os.path.isfile(_BUNDLE):
    # SSL_CERT_FILE is what OpenSSL reads in `set_default_verify_paths()` — the half of
    # `ssl.create_default_context()` that is not the Windows store, and therefore what
    # `urllib.request.urlopen` ends up verifying against. SSL_CERT_DIR is set beside it
    # so that neither half of OpenSSL's default pair is left pointing at the directory
    # the interpreter happened to be compiled for, which exists on nobody's machine.
    os.environ.setdefault("SSL_CERT_FILE", _BUNDLE)
    os.environ.setdefault("SSL_CERT_DIR", _CERTIFI)
