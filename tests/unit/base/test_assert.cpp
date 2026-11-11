#include <string>

#include <gtest/gtest.h>

#include "imsrg/base/assert.hpp"
#include "imsrg/base/error.hpp"

namespace {

using namespace imsrg;

TEST(Assert, CheckPassesSilently) {
  EXPECT_NO_THROW(IMSRG_CHECK(1 + 1 == 2));
  EXPECT_NO_THROW(IMSRG_CHECK(true, "never shown"));
}

TEST(Assert, CheckThrowsWithConditionFileAndMessage) {
  try {
    IMSRG_CHECK(1 + 1 == 3, "arithmetic is broken");
    FAIL() << "expected InvariantError";
  } catch (const InvariantError& e) {
    const std::string what = e.what();
    EXPECT_NE(what.find("1 + 1 == 3"), std::string::npos) << what;
    EXPECT_NE(what.find("test_assert.cpp"), std::string::npos) << what;
    EXPECT_NE(what.find("arithmetic is broken"), std::string::npos) << what;
  }
}

TEST(Assert, CheckWithoutMessage) {
  EXPECT_THROW(IMSRG_CHECK(false), InvariantError);
  EXPECT_THROW(IMSRG_CHECK(false), Error);
}

TEST(Assert, MessageNotEvaluatedWhenCheckPasses) {
  int evaluated = 0;
  auto msg = [&] {
    ++evaluated;
    return std::string("expensive");
  };
  IMSRG_CHECK(true, msg());
  EXPECT_EQ(evaluated, 0);
  EXPECT_THROW(IMSRG_CHECK(false, msg()), InvariantError);
  EXPECT_EQ(evaluated, 1);
}

TEST(Assert, AssertActiveInDebugBuild) {
#if IMSRG_ASSERTS_ENABLED
  EXPECT_THROW(IMSRG_ASSERT(false, "boom"), InvariantError);
  EXPECT_NO_THROW(IMSRG_ASSERT(true));
#else
  EXPECT_NO_THROW(IMSRG_ASSERT(false));
  GTEST_SKIP() << "assertions disabled in this build";
#endif
}

}  // namespace
