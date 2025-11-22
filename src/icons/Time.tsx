import { Clock as LucideClock, LucideProps } from 'lucide-react';

const Time = ({ className, ...props }: LucideProps) => {
  return <LucideClock className={className} {...props} />;
};

export default Time;