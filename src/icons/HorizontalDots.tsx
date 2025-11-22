import { MoreHorizontal as LucideMoreHorizontal, LucideProps } from 'lucide-react';

const HorizontalDots = ({ className, ...props }: LucideProps) => {
  return <LucideMoreHorizontal className={className} {...props} />;
};

export default HorizontalDots;