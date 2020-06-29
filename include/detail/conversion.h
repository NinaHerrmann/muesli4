/*
 * conversion.h
 *
 *      Author:
 *
 * -------------------------------------------------------------------------------
 *
 * The MIT License
 *
 * Copyright 2014 Steffen Ernsting <s.ernsting@uni-muenster.de>,
 *                Herbert Kuchen <kuchen@uni-muenster.de.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

#pragma once

namespace msl {

namespace detail {

// liefert true, wenn U oeffentlich von T erbt oder wenn T und U den gleichen Typ besitzen
#define MSL_IS_SUPERCLASS(T, U) (detail::Conversion<const U*, const T*>::exists && !detail::Conversion<const T*, const void*>::sameType)

/* MSL_Conversion erkennt zur Compilezeit, ob sich T nach U konvertieren laesst.
 exists = true, wenn sich T nach U konvertieren laesst, sonst false
 */
template<class T, class U>
class Conversion
{

private:

  // sizeof(Small) = 1
  typedef char Small;

  // sizeof(Big) > 1
  class Big
  {
    char dummy[2];
  };

  // Compiler waehlt diese Funktion, wenn er eine Umwandlung von T nach U findet
  static Small Test(U);
  // sonst nimmer er diese
  static Big Test(...);
  // Erzeugt ein Objekt vom Typ T, selbst wenn der Konstruktor als private deklariert wurde
  static T MakeT();

public:

  enum
  {
    exists = sizeof(Test(MakeT())) == sizeof(Small)
  };

  enum
  {
    sameType = false
  };

};

/* Überladung von MSL_Conversion, um den Fall T = U zu erkennen
 */
template<class T>
class Conversion<T, T>
{

public:

  enum
  {
    exists = true, sameType = true
  };

};

/* MSL_Int2Type erzeugt aus ganzzahligen Konstanten einen eigenen Typ. Wird
 benoetigt, damit der Compiler zur Compilezeit die korrekte MSL_Send Methode
 auswaehlen kann.
 */
template<int v>
struct Int2Type
{

  enum
  {
    value = v
  };

};

}

}
