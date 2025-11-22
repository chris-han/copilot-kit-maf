import { PieChart as LucidePieChart, LucideProps } from 'lucide-react';

const PieChart = ({ className, ...props }: LucideProps) => {
  return <LucidePieChart className={className} {...props} />;
};

export default PieChart;