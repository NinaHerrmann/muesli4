Muesli - Contribution guide
======
Contribution guide of the Muenster Skeleton Library (Muesli). It covers the branch structure and the style guide.


Branch structure
-----------------

This project follows the *anti-gitflow* pattern [blogpost](http://endoflineblog.com/gitflow-considered-harmful).

*master* is the eternal development branch.

Features are developed in feature branches starting with *feature/*. After completion, they are merged back into master and are deleted.
The same applies to release (*release/*) and hotfix (*hotfix/*) branches.
Release branches are used to prepare releases.
Hotfix branches are used to fix bugs in a already released version. The hotfix branch is created from the release tag.

Versions are marked with tags. Tags are placed on top of a release or hotfix branch. Then the branch is merged back into master and is deleted. The version of muesli is typically X.Y.Z. A new feature increases Y, a hotfix increases Z.



Style guide
------

This project follows the *anti-gitflow* pattern [Google C++ Style guide](https://google.github.io/styleguide/cppguide.html).

Unfortunately, this is not always the case, yet. So even if you see code that does not follow the style guide, consider breaking consistency and use the Google recommendations. And of course also adjust the old code to follow the style guide :)
