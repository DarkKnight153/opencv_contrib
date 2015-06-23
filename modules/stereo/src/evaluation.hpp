#include "precomp.hpp"

namespace cv
{
	namespace stereo
	{
		class Evaluation
		{
		public:
		    static double PerformDetailedTest(Mat rezult);
			Evaluation(void);
			~Evaluation(void);
		};
	}
}