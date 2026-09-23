# Security

## Reporting

Please report a vulnerability through GitHub's private reporting form
(Security tab, "Report a vulnerability"), not in a public issue. Expect a reply within a week.
Only the latest release is supported.

## What the code defends against

| Input | Risk | Defence |
| --- | --- | --- |
| Image files | disguised formats, decompression bombs, crashes in rarely used decoders | format allow-list read from the header (JPEG, PNG, WEBP, BMP), a 50-megapixel limit checked before decoding, every failure turned into a skipped file with a reason |
| Class directory names | path tricks, shell metacharacters in later tooling | strict pattern, at most 64 characters |
| Downloaded archive | tampering, path traversal, zip bombs | https only on every hop including redirects, pinned SHA-256 and size, member paths never joined to the destination, limits on files and on bytes actually written |
| Model files | a `.keras` file can carry Python code in a Lambda layer | models are loaded with `safe_mode=True`, and a test proves such a file is refused |

The pixel limit is a ceiling, not a dial. Pillow runs its own decompression-bomb check
(`Image.MAX_IMAGE_PIXELS`, 89,478,485 pixels here) while reading the same header,
before `load_rgb` looks at its own `max_pixels`, so `max_pixels` can tighten the limit but
cannot raise it past Pillow's default. Both bands of Pillow's check, the one that raises and
the one that only warns, are turned into the same refusal.

Even so, treat a model file like a program: load only files you trained or trust.

## What is out of scope

This is a command-line tool that runs with the privileges of its user on data the user points it
at. It opens no network port, stores no credentials and sends nothing anywhere; its only network
access is the EuroSAT download from Zenodo, on request.

## Supply chain

`uv.lock` pins every package with hashes. CI runs `pip-audit` and gitleaks on every push and
pull request and again weekly, GitHub Actions are pinned to a full commit SHA, with the release
tag in a comment, and Dependabot keeps both current, and the workflow token is read-only.
